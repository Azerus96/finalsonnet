from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
import utils
import github_utils
import time
import json
from threading import Thread, Event
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 час

# Глобальный экземпляр AI агента
cfr_agent = None

class GameManager:
    def __init__(self):
        self.active_games: Dict[str, Dict[str, Any]] = {}
        self.game_statistics: Dict[str, list] = {}
        
    def create_game(self, session_id: str) -> None:
        """Создает новую игру для сессии."""
        self.active_games[session_id] = {
            'state': self._create_initial_state(),
            'start_time': datetime.now(),
            'moves': [],
            'statistics': {
                'total_moves': 0,
                'fantasies': 0,
                'fouls': 0,
                'royalties': 0
            }
        }
        logger.info(f"Created new game for session {session_id}")

    def _create_initial_state(self) -> Dict[str, Any]:
        """Создает начальное состояние игры."""
        return {
            'selected_cards': [],
            'board': {'top': [], 'middle': [], 'bottom': []},
            'discarded_cards': [],
            'ai_settings': {
                'fantasyType': 'normal',
                'fantasyMode': False,
                'aiTime': '5',
                'iterations': '1000',
                'stopThreshold': '0.001',
                'aiType': 'mccfr'
            }
        }

    def update_game_state(self, session_id: str, new_state: Dict[str, Any]) -> None:
        """Обновляет состояние игры."""
        if session_id in self.active_games:
            self.active_games[session_id]['state'] = new_state
            self.active_games[session_id]['moves'].append({
                'timestamp': datetime.now(),
                'state': new_state.copy()
            })
            self._update_statistics(session_id, new_state)
            logger.debug(f"Updated game state for session {session_id}")
        else:
            logger.warning(f"Attempted to update non-existent game: {session_id}")

    def _update_statistics(self, session_id: str, state: Dict[str, Any]) -> None:
        """Обновляет статистику игры."""
        stats = self.active_games[session_id]['statistics']
        stats['total_moves'] += 1
        
        # Проверяем фантазию
        if self._check_fantasy(state):
            stats['fantasies'] += 1
        
        # Проверяем фол
        if self._check_foul(state):
            stats['fouls'] += 1
        
        # Подсчитываем роялти
        royalties = self._calculate_royalties(state)
        stats['royalties'] += royalties

    def _check_fantasy(self, state: Dict[str, Any]) -> bool:
        """Проверяет возможность фантазии."""
        top_cards = state['board']['top']
        if len(top_cards) >= 2:
            # Проверяем наличие пары дам или выше
            ranks = [card['rank'] for card in top_cards]
            high_pairs = ['Q', 'K', 'A']
            for rank in high_pairs:
                if ranks.count(rank) >= 2:
                    return True
        return False

    def _check_foul(self, state: Dict[str, Any]) -> bool:
        """Проверяет наличие фола."""
        if not all(len(state['board'][line]) > 0 for line in ['top', 'middle', 'bottom']):
            return False

        game_state = self._convert_to_game_state(state)
        return game_state.is_dead_hand()

    def _calculate_royalties(self, state: Dict[str, Any]) -> int:
        """Подсчитывает роялти."""
        game_state = self._convert_to_game_state(state)
        return int(game_state.calculate_score())

    def _convert_to_game_state(self, state: Dict[str, Any]) -> ai_engine.GameState:
        """Конвертирует состояние в объект GameState."""
        selected_cards = [ai_engine.Card(c['rank'], c['suit']) 
                         for c in state['selected_cards']]
        
        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            for card in state['board'][line]:
                board.place_card(line, ai_engine.Card(card['rank'], card['suit']))
        
        discarded_cards = [ai_engine.Card(c['rank'], c['suit']) 
                          for c in state['discarded_cards']]
        
        return ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=state['ai_settings']
        )

    def get_game_statistics(self, session_id: str) -> Dict[str, Any]:
        """Возвращает статистику игры."""
        if session_id in self.active_games:
            return self.active_games[session_id]['statistics']
        return {}

    def end_game(self, session_id: str) -> None:
        """Завершает игру и сохраняет статистику."""
        if session_id in self.active_games:
            game_data = self.active_games[session_id]
            game_data['end_time'] = datetime.now()
            self.game_statistics[session_id] = game_data
            del self.active_games[session_id]
            logger.info(f"Game ended for session {session_id}")

# Создаем экземпляр GameManager
game_manager = GameManager()
def initialize_ai_agent(ai_settings: Dict[str, Any]) -> None:
    """Инициализирует AI агента с заданными настройками."""
    global cfr_agent
    try:
        iterations = int(ai_settings.get('iterations', 1000))
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
        cfr_agent = ai_engine.CFRAgent(iterations=iterations, stop_threshold=stop_threshold)

        if os.environ.get("AI_PROGRESS_TOKEN"):
            try:
                cfr_agent.load_progress()
                logger.info("AI progress loaded successfully")
            except Exception as e:
                logger.error(f"Error loading AI progress: {e}")
        else:
            logger.warning("AI_PROGRESS_TOKEN not set. Progress loading disabled.")
            
    except Exception as e:
        logger.error(f"Error initializing AI agent: {e}")
        raise

def validate_game_state(state: Dict[str, Any]) -> bool:
    """Проверяет корректность состояния игры."""
    try:
        # Проверка структуры состояния
        required_keys = {'selected_cards', 'board', 'discarded_cards', 'ai_settings'}
        if not all(key in state for key in required_keys):
            return False

        # Проверка структуры доски
        if not all(line in state['board'] for line in ['top', 'middle', 'bottom']):
            return False

        # Проверка корректности карт
        for cards in [state['selected_cards'], state['discarded_cards']]:
            for card in cards:
                if not {'rank', 'suit'}.issubset(card.keys()):
                    return False
                if card['rank'] not in ai_engine.Card.RANKS:
                    return False
                if card['suit'] not in ai_engine.Card.SUITS:
                    return False

        # Проверка количества карт
        if len(state['board']['top']) > 3:
            return False
        if len(state['board']['middle']) > 5:
            return False
        if len(state['board']['bottom']) > 5:
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating game state: {e}")
        return False

@app.route('/')
def home():
    """Главная страница."""
    session_id = session.get('session_id')
    if not session_id:
        session['session_id'] = os.urandom(16).hex()
        game_manager.create_game(session['session_id'])
    return render_template('index.html')

@app.route('/training')
def training():
    """Страница тренировки."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = os.urandom(16).hex()
            game_manager.create_game(session['session_id'])
        
        if 'game_state' not in session:
            session['game_state'] = game_manager.active_games[session['session_id']]['state']

        # Инициализация AI агента при необходимости
        if cfr_agent is None or session['game_state']['ai_settings'] != session.get('previous_ai_settings'):
            initialize_ai_agent(session['game_state']['ai_settings'])
            session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()

        return render_template('training.html', game_state=session['game_state'])
    except Exception as e:
        logger.error(f"Error in training route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update_state', methods=['POST'])
def update_state():
    """Обновляет состояние игры."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content type must be application/json'}), 400

        game_state = request.get_json()
        
        if not validate_game_state(game_state):
            return jsonify({'error': 'Invalid game state format'}), 400

        session['game_state'] = game_state
        session.modified = True

        # Обновляем состояние в GameManager
        game_manager.update_game_state(session['session_id'], game_state)

        # Обновляем настройки AI при необходимости
        if game_state['ai_settings'] != session.get('previous_ai_settings'):
            initialize_ai_agent(game_state['ai_settings'])
            session['previous_ai_settings'] = game_state['ai_settings'].copy()

        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error in update_state: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Обрабатывает ход AI."""
    try:
        if cfr_agent is None:
            return jsonify({'error': 'AI agent not initialized'}), 500

        game_state_data = request.get_json()
        logger.debug(f"Received game_state_data: {game_state_data}")

        # Проверяем корректность входных данных
        if not validate_game_state(game_state_data):
            return jsonify({'error': 'Invalid game state format'}), 400

        # Конвертируем данные в объекты Card
        try:
            selected_cards = [ai_engine.Card(card['rank'], card['suit']) 
                            for card in game_state_data['selected_cards']]
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing selected cards: {e}")
            return jsonify({'error': 'Invalid card format'}), 400
          # Создаем объект Board и размещаем карты
        try:
            board = ai_engine.Board()
            for line in ['top', 'middle', 'bottom']:
                for card_data in game_state_data['board'].get(line, []):
                    board.place_card(line, ai_engine.Card(card_data['rank'], card_data['suit']))
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error processing board: {e}")
            return jsonify({'error': 'Invalid board format'}), 400

        # Обрабатываем сброшенные карты
        try:
            discarded_cards = [ai_engine.Card(card['rank'], card['suit']) 
                             for card in game_state_data.get('discarded_cards', [])]
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing discarded cards: {e}")
            return jsonify({'error': 'Invalid discarded cards format'}), 400

        # Создаем состояние игры
        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=game_state_data['ai_settings']
        )

        # Получаем ход AI
        timeout_event = Event()
        result = {'move': None}
        num_cards = len(selected_cards)

        ai_thread = Thread(
            target=cfr_agent.get_move,
            args=(game_state, num_cards, timeout_event, result)
        )
        ai_thread.start()

        # Ждем завершения работы AI
        ai_time = int(game_state_data['ai_settings'].get('aiTime', 5))
        ai_thread.join(timeout=ai_time)

        if ai_thread.is_alive():
            timeout_event.set()
            ai_thread.join()
            logger.warning("AI move timed out")
            return jsonify({'error': 'AI move timed out'})

        if 'error' in result.get('move', {}):
            error_message = result['move'].get('error', 'Unknown error in AI move')
            logger.error(f"Error in AI move: {error_message}")
            return jsonify({'error': error_message}), 500

        move = result['move']

        # Сериализуем ход AI
        def serialize_card(card):
            return card.to_dict() if card is not None else None

        def serialize_move(move):
            serialized = {}
            for key, cards in move.items():
                if cards is not None:
                    serialized[key] = [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
                else:
                    serialized[key] = None
            return serialized

        serialized_move = serialize_move(move)

        # Обновляем состояние игры
        if move:
            session['game_state']['board'] = {
                'top': [serialize_card(card) for card in move.get('top', [])],
                'middle': [serialize_card(card) for card in move.get('middle', [])],
                'bottom': [serialize_card(card) for card in move.get('bottom', [])]
            }
            
            if move.get('discarded'):
                discarded = move['discarded']
                if isinstance(discarded, list):
                    session['game_state']['discarded_cards'].extend([serialize_card(card) for card in discarded])
                else:
                    session['game_state']['discarded_cards'].append(serialize_card(discarded))
            
            session.modified = True
            game_manager.update_game_state(session['session_id'], session['game_state'])

        # Сохраняем прогресс AI
        try:
            if cfr_agent.iterations % 100 == 0:
                cfr_agent.save_progress()
                logger.info("AI progress saved successfully")
        except Exception as e:
            logger.error(f"Error saving AI progress: {e}")

        return jsonify(serialized_move)

    except Exception as e:
        logger.error(f"Unexpected error in ai_move: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_statistics')
def get_statistics():
    """Возвращает статистику игры."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        statistics = game_manager.get_game_statistics(session_id)
        return jsonify(statistics)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/end_game', methods=['POST'])
def end_game():
    """Завершает текущую игру."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        game_manager.end_game(session_id)
        session.clear()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error ending game: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Обработчик ошибки 404."""
    logger.error(f"404 error: {error}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработчик ошибки 500."""
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Инициализируем AI агента с настройками по умолчанию
    initialize_ai_agent({
        'iterations': 1000,
        'stopThreshold': 0.001,
        'aiTime': 5,
        'aiType': 'mccfr'
    })
    
    app.run(debug=True)

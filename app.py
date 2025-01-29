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

# Создаем необходимые директории
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('flask_session'):
    os.makedirs('flask_session')

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('FLASK_ENV') == 'development' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Проверка окружения
if not os.environ.get("SECRET_KEY"):
    logger.warning("SECRET_KEY not set, generating random key")
    os.environ["SECRET_KEY"] = os.urandom(24).hex()

if not os.environ.get("AI_PROGRESS_TOKEN"):
    logger.warning("AI_PROGRESS_TOKEN not set")

# Настройка Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Глобальный экземпляр AI агента
cfr_agent = None

class GameManager:
    def __init__(self):
        self.active_games: Dict[str, Dict[str, Any]] = {}
        self.game_statistics: Dict[str, list] = {}

    def create_game(self, session_id: str) -> None:
        """Создает новую игру для сессии."""
        try:
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
        except Exception as e:
            logger.error(f"Error creating game for session {session_id}: {e}")
            raise

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
        try:
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
        except Exception as e:
            logger.error(f"Error updating game state: {e}")
            raise

    def _update_statistics(self, session_id: str, state: Dict[str, Any]) -> None:
        """Обновляет статистику игры."""
        try:
            stats = self.active_games[session_id]['statistics']
            stats['total_moves'] += 1
            
            if self._check_fantasy(state):
                stats['fantasies'] += 1
            if self._check_foul(state):
                stats['fouls'] += 1
            
            royalties = self._calculate_royalties(state)
            stats['royalties'] += royalties
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def _check_fantasy(self, state: Dict[str, Any]) -> bool:
        """Проверяет возможность фантазии."""
        try:
            top_cards = state['board']['top']
            if len(top_cards) >= 2:
                ranks = [card['rank'] for card in top_cards]
                high_pairs = ['Q', 'K', 'A']
                for rank in high_pairs:
                    if ranks.count(rank) >= 2:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking fantasy: {e}")
            return False

    def _check_foul(self, state: Dict[str, Any]) -> bool:
        """Проверяет наличие фола."""
        try:
            if not all(len(state['board'][line]) > 0 for line in ['top', 'middle', 'bottom']):
                return False

            game_state = self._convert_to_game_state(state)
            return game_state.is_dead_hand()
        except Exception as e:
            logger.error(f"Error checking foul: {e}")
            return False

    def _calculate_royalties(self, state: Dict[str, Any]) -> int:
        """Подсчитывает роялти."""
        try:
            game_state = self._convert_to_game_state(state)
            return int(game_state.calculate_score())
        except Exception as e:
            logger.error(f"Error calculating royalties: {e}")
            return 0

    def _convert_to_game_state(self, state: Dict[str, Any]) -> ai_engine.GameState:
        """Конвертирует состояние в объект GameState."""
        try:
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
        except Exception as e:
            logger.error(f"Error converting to game state: {e}")
            raise

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
    """Инициализация AI агента с обработкой ошибок"""
    global cfr_agent
    try:
        iterations = int(ai_settings.get('iterations', 1000))
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
        cfr_agent = ai_engine.CFRAgent(
            iterations=iterations,
            stop_threshold=stop_threshold
        )
        logger.info("AI agent initialized successfully")

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
        logger.info("Falling back to default AI agent settings")
        cfr_agent = ai_engine.CFRAgent()

def validate_game_state(state: Dict[str, Any]) -> bool:
    """Проверяет корректность состояния игры."""
    try:
        # Проверка структуры состояния
        required_keys = {'selected_cards', 'board', 'discarded_cards', 'ai_settings'}
        if not all(key in state for key in required_keys):
            logger.error("Missing required keys in game state")
            return False

        # Проверка структуры доски
        if not all(line in state['board'] for line in ['top', 'middle', 'bottom']):
            logger.error("Invalid board structure")
            return False

        # Проверка корректности карт
        for cards in [state['selected_cards'], state['discarded_cards']]:
            for card in cards:
                if not {'rank', 'suit'}.issubset(card.keys()):
                    logger.error(f"Invalid card structure: {card}")
                    return False
                if card['rank'] not in ai_engine.Card.RANKS:
                    logger.error(f"Invalid card rank: {card['rank']}")
                    return False
                if card['suit'] not in ai_engine.Card.SUITS:
                    logger.error(f"Invalid card suit: {card['suit']}")
                    return False

        # Проверка количества карт
        if len(state['board']['top']) > 3:
            logger.error("Too many cards in top row")
            return False
        if len(state['board']['middle']) > 5:
            logger.error("Too many cards in middle row")
            return False
        if len(state['board']['bottom']) > 5:
            logger.error("Too many cards in bottom row")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating game state: {e}")
        return False

# Обработка ошибок
@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

# Настройка сессии
@app.before_request
def before_request():
    """Настройка сессии перед каждым запросом"""
    if not session.get('session_id'):
        session['session_id'] = os.urandom(16).hex()
        logger.info(f"Created new session: {session['session_id']}")

@app.route('/')
def home():
    """Главная страница."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = os.urandom(16).hex()
            game_manager.create_game(session['session_id'])
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/training')
def training():
    """Страница тренировки."""
    try:
        session_id = session.get('session_id')
        
        if not session_id:
            session['session_id'] = os.urandom(16).hex()
            game_manager.create_game(session['session_id'])
        
        # Проверяем, существует ли игра для данной сессии
        if session_id not in game_manager.active_games:
            logger.error(f"Game not found for session: {session_id}")
            return jsonify({'error': 'Game not found for this session'}), 404
        
        # Получаем состояние игры
        session_game_state = game_manager.active_games[session_id]['state']
        session['game_state'] = session_game_state  # Двигайтесь по актуальному состоянию

        # Инициализация AI агента при необходимости
        if cfr_agent is None or session_game_state['ai_settings'] != session.get('previous_ai_settings'):
            initialize_ai_agent(session_game_state['ai_settings'])
            session['previous_ai_settings'] = session_game_state['ai_settings'].copy()

        return render_template('training.html', game_state=session_game_state)
    
    except Exception as e:
        logger.error(f"Error in training route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update_state', methods=['POST'])
def update_state():
    """Обновляет состояние игры."""
    try:
        if not request.is_json:
            logger.warning("Received non-JSON request")
            return jsonify({'error': 'Content type must be application/json'}), 400

        game_state = request.get_json()
        
        if not validate_game_state(game_state):
            logger.warning("Invalid game state received")
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
            logger.error("AI agent not initialized")
            return jsonify({'error': 'AI agent not initialized'}), 500

        game_state_data = request.get_json()
        logger.debug(f"Received game state data for AI move")

        if not validate_game_state(game_state_data):
            logger.warning("Invalid game state received for AI move")
            return jsonify({'error': 'Invalid game state format'}), 400

        # Конвертируем данные в объекты Card
        try:
            selected_cards = [ai_engine.Card(card['rank'], card['suit']) 
                              for card in game_state_data['selected_cards']]
            
            board = ai_engine.Board()
            for line in ['top', 'middle', 'bottom']:
                for card_data in game_state_data['board'].get(line, []):
                    board.place_card(line, ai_engine.Card(card_data['rank'], card_data['suit']))
            
            discarded_cards = [ai_engine.Card(card['rank'], card['suit']) 
                               for card in game_state_data.get('discarded_cards', [])]
            
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

            if result.get('move') is None:
                logger.error("AI failed to produce a move")
                return jsonify({'error': 'AI failed to produce a move'}), 500

            move = result['move']

            # Сериализуем ход AI
            def serialize_card(card):
                return card.to_dict() if card is not None else None

            def serialize_move(move):
                serialized = {}
                for key, value in move.items():
                    if value is not None:
                        if isinstance(value, list):
                            serialized[key] = [serialize_card(card) for card in value]
                        else:
                            serialized[key] = serialize_card(value)
                    else:
                        serialized[key] = None
                return serialized

            serialized_move = serialize_move(move)
            logger.info("AI move generated successfully")
            return jsonify(serialized_move)

        except Exception as e:
            logger.error(f"Error processing AI move: {e}")
            return jsonify({'error': str(e)}), 500

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

# Запуск приложения
if __name__ == '__main__':
    # Создаем необходимые директории
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('flask_session'):
        os.makedirs('flask_session')

    # Инициализируем AI агента с настройками по умолчанию
    initialize_ai_agent({
        'iterations': 1000,
        'stopThreshold': 0.001,
        'aiTime': 5,
        'aiType': 'mccfr'
    })
    
    # Получаем порт из переменных окружения или используем порт по умолчанию
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)

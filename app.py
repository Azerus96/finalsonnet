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
        try:
            if session_id in self.active_games:
                return self.active_games[session_id]['statistics']
            return {}
        except Exception as e:
            logger.error(f"Error getting game statistics: {e}")
            return {}

    def end_game(self, session_id: str) -> None:
        """Завершает игру и сохраняет статистику."""
        try:
            if session_id in self.active_games:
                game_data = self.active_games[session_id]
                game_data['end_time'] = datetime.now()
                self.game_statistics[session_id] = game_data
                del self.active_games[session_id]
                logger.info(f"Game ended for session {session_id}")
        except Exception as e:
            logger.error(f"Error ending game: {e}")

# Создаем экземпляр GameManager
game_manager = GameManager()

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

@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    """Получает текущее состояние игры."""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in game_manager.active_games:
            return jsonify({'error': 'Invalid session'}), 400
        
        return jsonify(game_manager.active_games[session_id]['state'])
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/game/move', methods=['POST'])
def make_move():
    """Обрабатывает ход игрока."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 400

        move_data = request.get_json()
        if not move_data:
            return jsonify({'error': 'Invalid move data'}), 400

        if not validate_game_state(move_data):
            return jsonify({'error': 'Invalid game state'}), 400

        game_manager.update_game_state(session_id, move_data)
        
        return jsonify({
            'status': 'success',
            'statistics': game_manager.get_game_statistics(session_id)
        })
    except Exception as e:
        logger.error(f"Error processing move: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/game/end', methods=['POST'])
def end_game():
    """Завершает текущую игру."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 400

        game_manager.end_game(session_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error ending game: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Получает статистику игры."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 400

        statistics = game_manager.get_game_statistics(session_id)
        return jsonify(statistics)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)

import unittest
import sys
import os
from pathlib import Path
import json
import pickle
from unittest.mock import Mock, patch
from datetime import datetime
import threading
import time

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from ai_engine import Card, Hand, Board, GameState, CFRAgent
from utils import DataManager, StatisticsManager
from github_utils import GitHubSync

class TestCard(unittest.TestCase):
    """Тесты для класса Card."""

    def test_card_creation(self):
        """Тест создания карты."""
        card = Card('A', '♠')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, '♠')

    def test_invalid_card(self):
        """Тест создания карты с неверными параметрами."""
        with self.assertRaises(ValueError):
            Card('Z', '♠')
        with self.assertRaises(ValueError):
            Card('A', 'X')

    def test_card_comparison(self):
        """Тест сравнения карт."""
        card1 = Card('A', '♠')
        card2 = Card('A', '♠')
        card3 = Card('K', '♠')
        self.assertEqual(card1, card2)
        self.assertNotEqual(card1, card3)

class TestHand(unittest.TestCase):
    """Тесты для класса Hand."""

    def setUp(self):
        self.card1 = Card('A', '♠')
        self.card2 = Card('K', '♠')
        self.hand = Hand([self.card1, self.card2])

    def test_hand_creation(self):
        """Тест создания руки."""
        self.assertEqual(len(self.hand), 2)
        self.assertIn(self.card1, self.hand.cards)

    def test_add_remove_card(self):
        """Тест добавления и удаления карт."""
        card3 = Card('Q', '♠')
        self.hand.add_card(card3)
        self.assertEqual(len(self.hand), 3)
        self.hand.remove_card(card3)
        self.assertEqual(len(self.hand), 2)

class TestBoard(unittest.TestCase):
    """Тесты для класса Board."""

    def setUp(self):
        self.board = Board()
        self.card = Card('A', '♠')

    def test_place_card(self):
        """Тест размещения карты на доске."""
        self.board.place_card('top', self.card)
        self.assertIn(self.card, self.board.top)

    def test_line_limits(self):
        """Тест ограничений на количество карт в линиях."""
        for _ in range(3):
            self.board.place_card('top', Card('A', '♠'))
        with self.assertRaises(ValueError):
            self.board.place_card('top', Card('K', '♠'))

    def test_is_full(self):
        """Тест проверки заполненности доски."""
        self.assertFalse(self.board.is_full())
        
        # Заполняем доску
        for _ in range(3):
            self.board.place_card('top', Card('A', '♠'))
        for _ in range(5):
            self.board.place_card('middle', Card('K', '♠'))
        for _ in range(5):
            self.board.place_card('bottom', Card('Q', '♠'))
            
        self.assertTrue(self.board.is_full())

class TestGameState(unittest.TestCase):
    """Тесты для класса GameState."""

    def setUp(self):
        self.game_state = GameState()

    def test_initial_state(self):
        """Тест начального состояния игры."""
        self.assertEqual(len(self.game_state.selected_cards), 0)
        self.assertFalse(self.game_state.board.is_full())

    def test_dead_hand(self):
        """Тест определения мертвой руки."""
        # Создаем ситуацию с мертвой рукой (верхняя линия сильнее средней)
        self.game_state.board.place_card('top', Card('A', '♠'))
        self.game_state.board.place_card('top', Card('A', '♥'))
        self.game_state.board.place_card('top', Card('A', '♦'))
        
        self.game_state.board.place_card('middle', Card('K', '♠'))
        self.game_state.board.place_card('middle', Card('K', '♥'))
        self.game_state.board.place_card('middle', Card('Q', '♦'))
        self.game_state.board.place_card('middle', Card('J', '♠'))
        self.game_state.board.place_card('middle', Card('10', '♥'))

        self.assertTrue(self.game_state.is_dead_hand())

class TestCFRAgent(unittest.TestCase):
    """Тесты для класса CFRAgent."""

    def setUp(self):
        self.agent = CFRAgent(iterations=100)

    def test_agent_initialization(self):
        """Тест инициализации агента."""
        self.assertEqual(self.agent.iterations, 100)
        self.assertEqual(len(self.agent.nodes), 0)

    @patch('threading.Event')
    def test_training(self, mock_event):
        """Тест процесса обучения."""
        mock_event.return_value = threading.Event()
        result = {}
        self.agent.train(mock_event(), result)
        self.assertGreater(len(self.agent.nodes), 0)

class TestDataManager(unittest.TestCase):
    """Тесты для класса DataManager."""

    def setUp(self):
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        self.data_manager = DataManager(str(self.test_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_save_load_pickle(self):
        """Тест сохранения и загрузки данных в формате pickle."""
        test_data = {'test': 'data'}
        self.data_manager.save_data(test_data, 'test.pkl')
        loaded_data = self.data_manager.load_data('test.pkl')
        self.assertEqual(test_data, loaded_data)

    def test_save_load_json(self):
        """Тест сохранения и загрузки данных в формате JSON."""
        test_data = {'test': 'data'}
        self.data_manager.save_data(test_data, 'test.json')
        loaded_data = self.data_manager.load_data('test.json')
        self.assertEqual(test_data, loaded_data)

class TestGitHubSync(unittest.TestCase):
    """Тесты для класса GitHubSync."""

    @patch('github.Github')
    def setUp(self, mock_github):
        self.github_sync = GitHubSync()
        self.mock_github = mock_github

    def test_sync_interval(self):
        """Тест интервала синхронизации."""
        filename = 'test.json'
        self.assertTrue(self.github_sync._should_sync(filename))
        self.github_sync._last_sync[filename] = time.time()
        self.assertFalse(self.github_sync._should_sync(filename))

def run_tests():
    """Запускает все тесты."""
    unittest.main(argv=[''], verbosity=2)

if __name__ == '__main__':
    run_tests()

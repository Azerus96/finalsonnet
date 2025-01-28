import random
import itertools
from collections import defaultdict
from threading import Thread, Event
import time
import pickle
from typing import List, Dict, Set, Optional, Tuple

class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']
    
    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def to_dict(self) -> Dict[str, str]:
        return {'rank': self.rank, 'suit': self.suit}

    @property
    def rank_value(self) -> int:
        return self.RANKS.index(self.rank)

    @classmethod
    def get_all_cards(cls) -> List['Card']:
        return [Card(rank, suit) for rank in cls.RANKS for suit in cls.SUITS]

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Card':
        return cls(data['rank'], data['suit'])

class Hand:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards is not None else []

    def add_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.remove(card)

    def get_ranks(self) -> List[str]:
        return [card.rank for card in self.cards]

    def get_suits(self) -> List[str]:
        return [card.suit for card in self.cards]

    def __repr__(self) -> str:
        return ', '.join(map(str, self.cards))

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]

    def clear(self) -> None:
        self.cards.clear()

class Board:
    def __init__(self):
        self.top: List[Card] = []
        self.middle: List[Card] = []
        self.bottom: List[Card] = []
        self.max_cards = {'top': 3, 'middle': 5, 'bottom': 5}

    def place_card(self, line: str, card: Card) -> None:
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")
        
        target_line = getattr(self, line)
        if len(target_line) >= self.max_cards[line]:
            raise ValueError(f"{line.capitalize()} line is full")
        
        target_line.append(card)

    def remove_card(self, line: str, card: Card) -> None:
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError(f"Invalid line: {line}")
        target_line = getattr(self, line)
        target_line.remove(card)

    def get_line(self, line: str) -> List[Card]:
        if line not in ['top', 'middle', 'bottom']:
            raise ValueError(f"Invalid line: {line}")
        return getattr(self, line).copy()

    def is_full(self) -> bool:
        return (len(self.top) == self.max_cards['top'] and 
                len(self.middle) == self.max_cards['middle'] and 
                len(self.bottom) == self.max_cards['bottom'])

    def clear(self) -> None:
        self.top.clear()
        self.middle.clear()
        self.bottom.clear()

    def __repr__(self) -> str:
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            'top': [card.to_dict() for card in self.top],
            'middle': [card.to_dict() for card in self.middle],
            'bottom': [card.to_dict() for card in self.bottom]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, str]]]) -> 'Board':
        board = cls()
        for line in ['top', 'middle', 'bottom']:
            for card_data in data.get(line, []):
                board.place_card(line, Card.from_dict(card_data))
        return board
class GameState:
    def __init__(self, selected_cards: Optional[List[Card]] = None,
                 board: Optional[Board] = None,
                 discarded_cards: Optional[List[Card]] = None,
                 ai_settings: Optional[Dict] = None):
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0
        self.remaining_deck = self._initialize_deck()

    def _initialize_deck(self) -> Set[Card]:
        """Инициализирует колоду с учетом уже использованных карт."""
        all_cards = set(Card.get_all_cards())
        used_cards = set(self.selected_cards.cards + 
                        self.board.top + 
                        self.board.middle + 
                        self.board.bottom + 
                        self.discarded_cards)
        return all_cards - used_cards

    def get_current_player(self) -> int:
        return self.current_player

    def get_actions(self) -> List[Dict]:
        """Возвращает все возможные действия в текущем состоянии."""
        num_cards = len(self.selected_cards)
        actions = []

        if num_cards == 5:  # Первая улица
            for p in itertools.permutations(self.selected_cards.cards):
                actions.append({
                    'top': [p[0]],
                    'middle': [p[1], p[2]],
                    'bottom': [p[3], p[4]],
                    'discarded': None
                })
        elif num_cards == 3:  # Последующие улицы
            for p in itertools.permutations(self.selected_cards.cards, 2):
                for discard in self.selected_cards.cards:
                    if discard not in p:
                        # Генерируем все возможные комбинации размещения двух карт
                        possible_placements = [
                            {'top': [p[0]], 'middle': [p[1]], 'bottom': [], 'discarded': discard},
                            {'top': [p[0]], 'middle': [], 'bottom': [p[1]], 'discarded': discard},
                            {'top': [], 'middle': [p[0]], 'bottom': [p[1]], 'discarded': discard},
                            {'top': [], 'middle': [p[1]], 'bottom': [p[0]], 'discarded': discard}
                        ]
                        actions.extend(possible_placements)

        elif num_cards >= 13:  # Режим фантазии
            fantasy_cards = self.selected_cards.cards[:13]
            discarded_cards = self.selected_cards.cards[13:]
            
            for p in itertools.permutations(fantasy_cards):
                actions.append({
                    'top': list(p[:3]),
                    'middle': list(p[3:8]),
                    'bottom': list(p[8:13]),
                    'discarded': discarded_cards
                })

        return actions

    def apply_action(self, action: Dict) -> 'GameState':
        """Применяет действие к текущему состоянию и возвращает новое состояние."""
        new_board = Board()
        new_discarded = self.discarded_cards.copy()

        # Размещаем карты на доске
        for line in ['top', 'middle', 'bottom']:
            if action.get(line):
                for card in action[line]:
                    new_board.place_card(line, card)

        # Обрабатываем сброшенные карты
        if action.get('discarded'):
            if isinstance(action['discarded'], list):
                new_discarded.extend(action['discarded'])
            else:
                new_discarded.append(action['discarded'])

        # Создаем новое состояние
        new_state = GameState(
            selected_cards=[],  # Очищаем выбранные карты
            board=new_board,
            discarded_cards=new_discarded,
            ai_settings=self.ai_settings
        )

        return new_state

    def is_terminal(self) -> bool:
        """Проверяет, является ли состояние терминальным."""
        return self.board.is_full()

    def get_information_set(self) -> str:
        """Возвращает строковое представление информационного набора."""
        board_info = []
        for line in ['top', 'middle', 'bottom']:
            cards = getattr(self.board, line)
            sorted_cards = sorted(cards, key=lambda c: (c.rank_value, Card.SUITS.index(c.suit)))
            board_info.append(f"{line}:{','.join(map(str, sorted_cards))}")
        
        discarded = ','.join(map(str, sorted(self.discarded_cards, 
                                           key=lambda c: (c.rank_value, Card.SUITS.index(c.suit)))))
        
        return f"{';'.join(board_info)}|D:{discarded}"

    def calculate_outs(self) -> List[Card]:
        """Рассчитывает возможные ауты для текущего состояния."""
        outs = []
        for card in self.remaining_deck:
            # Создаем копию состояния для тестирования карты
            test_state = self._create_test_state(card)
            if self._would_improve_hand(test_state):
                outs.append(card)
        return outs

    def _create_test_state(self, test_card: Card) -> 'GameState':
        """Создает тестовое состояние с добавленной картой."""
        test_selected = self.selected_cards.cards.copy()
        test_selected.append(test_card)
        return GameState(
            selected_cards=test_selected,
            board=self.board,
            discarded_cards=self.discarded_cards,
            ai_settings=self.ai_settings
        )

    def _would_improve_hand(self, test_state: 'GameState') -> bool:
        """Проверяет, улучшит ли тестовое состояние текущую руку."""
        current_score = self.calculate_score()
        test_score = test_state.calculate_score()
        return test_score > current_score
def calculate_score(self) -> float:
        """Рассчитывает общий счет для текущего состояния."""
        if self.is_dead_hand():
            return -1000.0

        score = 0.0
        
        # Базовые очки за комбинации
        score += self._get_line_score('top', self.board.top)
        score += self._get_line_score('middle', self.board.middle)
        score += self._get_line_score('bottom', self.board.bottom)

        # Бонус за фантазию
        if self.ai_settings.get('fantasyMode', False):
            score += self._calculate_fantasy_bonus()

        # Штраф за риск фола
        foul_risk = self._calculate_foul_risk()
        score -= foul_risk * 50  # Высокий штраф за риск фола

        # Бонус за потенциал улучшения
        improvement_potential = self._calculate_improvement_potential()
        score += improvement_potential * 10

        return score

    def _calculate_foul_risk(self) -> float:
        """Рассчитывает риск фола для текущего состояния."""
        risk = 0.0
        
        if len(self.board.top) > 0 and len(self.board.middle) > 0:
            top_strength = self._evaluate_hand_strength(self.board.top)
            middle_strength = self._evaluate_hand_strength(self.board.middle)
            if top_strength > middle_strength:
                risk += 0.5

        if len(self.board.middle) > 0 and len(self.board.bottom) > 0:
            middle_strength = self._evaluate_hand_strength(self.board.middle)
            bottom_strength = self._evaluate_hand_strength(self.board.bottom)
            if middle_strength > bottom_strength:
                risk += 0.5

        return risk

    def _calculate_improvement_potential(self) -> float:
        """Оценивает потенциал улучшения руки."""
        potential = 0.0
        outs = self.calculate_outs()
        
        if outs:
            # Вероятность получения аута
            probability = len(outs) / len(self.remaining_deck) if self.remaining_deck else 0
            potential = probability * self._evaluate_outs_value(outs)

        return potential

    def _evaluate_outs_value(self, outs: List[Card]) -> float:
        """Оценивает ценность возможных аутов."""
        value = 0.0
        for out in outs:
            # Оцениваем каждый аут на основе его потенциала для улучшения руки
            test_state = self._create_test_state(out)
            value += test_state.calculate_score() - self.calculate_score()
        return value / len(outs) if outs else 0

    def _calculate_fantasy_bonus(self) -> float:
        """Рассчитывает бонус за возможность фантазии."""
        if not self.board.top:
            return 0.0

        # Проверяем наличие высокой пары или сета в верхней линии
        top_ranks = [card.rank for card in self.board.top]
        if len(set(top_ranks)) == 1:  # Сет
            rank_value = Card.RANKS.index(top_ranks[0])
            if rank_value >= Card.RANKS.index('Q'):  # От дам и выше
                return 100.0
        elif len(set(top_ranks)) == 2:  # Пара
            pairs = [rank for rank in set(top_ranks) if top_ranks.count(rank) == 2]
            if pairs and Card.RANKS.index(pairs[0]) >= Card.RANKS.index('Q'):
                return 50.0

        return 0.0

    def _evaluate_hand_strength(self, cards: List[Card]) -> float:
        """Оценивает силу комбинации карт."""
        if not cards:
            return 0.0

        strength = 0.0
        
        # Базовая сила от старших карт
        for card in cards:
            strength += Card.RANKS.index(card.rank) * 0.1

        # Дополнительные очки за комбинации
        if len(cards) >= 3:
            if self._is_straight_flush(cards):
                strength += 100
            elif self._is_four_of_kind(cards):
                strength += 90
            elif self._is_full_house(cards):
                strength += 80
            elif self._is_flush(cards):
                strength += 70
            elif self._is_straight(cards):
                strength += 60
            elif self._is_three_of_kind(cards):
                strength += 50
            elif self._is_two_pair(cards):
                strength += 40
            elif self._is_one_pair(cards):
                strength += 30

        return strength

    def is_dead_hand(self) -> bool:
        """Проверяет, является ли рука мертвой."""
        if not self.board.is_full():
            return False

        top_strength = self._evaluate_hand_strength(self.board.top)
        middle_strength = self._evaluate_hand_strength(self.board.middle)
        bottom_strength = self._evaluate_hand_strength(self.board.bottom)

        return top_strength > middle_strength or middle_strength > bottom_strength

    def _get_line_score(self, line: str, cards: List[Card]) -> float:
        """Рассчитывает очки для конкретной линии."""
        if not cards:
            return 0.0

        score = 0.0
        
        # Базовые очки за комбинации
        if line == 'top' and len(cards) == 3:
            if self._is_three_of_kind(cards):
                score += self._get_set_bonus(cards[0].rank)
            elif self._is_one_pair(cards):
                score += self._get_pair_bonus(cards)
            else:
                score += self._get_high_card_bonus(cards)

        elif len(cards) == 5:  # middle или bottom
            if self._is_straight_flush(cards):
                score += 25 if line == 'bottom' else 50
            elif self._is_four_of_kind(cards):
                score += 10 if line == 'bottom' else 20
            elif self._is_full_house(cards):
                score += 6 if line == 'bottom' else 12
            elif self._is_flush(cards):
                score += 4 if line == 'bottom' else 8
            elif self._is_straight(cards):
                score += 2 if line == 'bottom' else 4

        return score
def _get_set_bonus(self, rank: str) -> float:
        """Возвращает бонус за сет в верхней линии."""
        bonus_values = {
            '2': 10, '3': 11, '4': 12, '5': 13, '6': 14,
            '7': 15, '8': 16, '9': 17, '10': 18,
            'J': 19, 'Q': 20, 'K': 21, 'A': 22
        }
        return float(bonus_values.get(rank, 0))

    def _get_pair_bonus(self, cards: List[Card]) -> float:
        """Возвращает бонус за пару в верхней линии."""
        ranks = [card.rank for card in cards]
        for rank in Card.RANKS[::-1]:
            if ranks.count(rank) == 2:
                pair_values = {
                    '6': 1, '7': 2, '8': 3, '9': 4, '10': 5,
                    'J': 6, 'Q': 7, 'K': 8, 'A': 9
                }
                return float(pair_values.get(rank, 0))
        return 0.0

    def _get_high_card_bonus(self, cards: List[Card]) -> float:
        """Возвращает бонус за старшую карту в верхней линии."""
        if len(set(card.rank for card in cards)) == 3:  # Все карты разные
            highest_card = max(cards, key=lambda c: Card.RANKS.index(c.rank))
            return 1.0 if highest_card.rank == 'A' else 0.0
        return 0.0

    # Методы проверки комбинаций
    def _is_straight_flush(self, cards: List[Card]) -> bool:
        return self._is_straight(cards) and self._is_flush(cards)

    def _is_four_of_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 4 for rank in set(ranks))

    def _is_full_house(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        return 3 in rank_counts.values() and 2 in rank_counts.values()

    def _is_flush(self, cards: List[Card]) -> bool:
        return len(set(card.suit for card in cards)) == 1

    def _is_straight(self, cards: List[Card]) -> bool:
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        
        # Проверка на обычный стрит
        if all(ranks[i+1] - ranks[i] == 1 for i in range(len(ranks)-1)):
            return True
            
        # Проверка на стрит с тузом внизу (A,2,3,4,5)
        if ranks == [0, 1, 2, 3, 12]:  # 2,3,4,5,A
            return True
            
        return False

    def _is_three_of_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 3 for rank in set(ranks))

    def _is_two_pair(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        pairs = [rank for rank in set(ranks) if ranks.count(rank) == 2]
        return len(pairs) == 2

    def _is_one_pair(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 2 for rank in set(ranks))

class CFRNode:
    def __init__(self, actions: List[Dict]):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions
        self.average_strategy = None
        self.current_strategy = None

    def get_strategy(self, realization_weight: float) -> Dict[Dict, float]:
        """Возвращает текущую стратегию на основе накопленных сожалений."""
        normalizing_sum = 0.0
        strategy = defaultdict(float)

        # Конвертируем действия в хешируемый формат для использования в качестве ключей
        hashable_actions = [self._action_to_hashable(a) for a in self.actions]

        for action in hashable_actions:
            strategy[action] = max(0, self.regret_sum[action])
            normalizing_sum += strategy[action]

        if normalizing_sum > 0:
            for action in hashable_actions:
                strategy[action] /= normalizing_sum
        else:
            # Если нет положительных сожалений, используем равномерное распределение
            prob = 1.0 / len(hashable_actions)
            for action in hashable_actions:
                strategy[action] = prob

        # Обновляем накопленную стратегию
        for action in hashable_actions:
            self.strategy_sum[action] += realization_weight * strategy[action]

        self.current_strategy = {self._hashable_to_action(k): v for k, v in strategy.items()}
        return self.current_strategy

    def get_average_strategy(self) -> Dict[Dict, float]:
        """Возвращает усредненную стратегию."""
        if self.average_strategy is not None:
            return self.average_strategy

        strategy = defaultdict(float)
        normalizing_sum = sum(self.strategy_sum.values())

        if normalizing_sum > 0:
            for action, sum_prob in self.strategy_sum.items():
                strategy[self._hashable_to_action(action)] = sum_prob / normalizing_sum
        else:
            # Равномерное распределение, если нет накопленной стратегии
            prob = 1.0 / len(self.actions)
            for action in self.actions:
                strategy[action] = prob

        self.average_strategy = strategy
        return strategy

    @staticmethod
    def _action_to_hashable(action: Dict) -> tuple:
        """Конвертирует действие в хешируемый формат."""
        result = []
        for key in sorted(action.keys()):
            if isinstance(action[key], list):
                result.append((key, tuple(str(card) for card in action[key])))
            elif action[key] is None:
                result.append((key, None))
            else:
                result.append((key, str(action[key])))
        return tuple(result)

    @staticmethod
    def _hashable_to_action(hashable_action: tuple) -> Dict:
        """Конвертирует хешируемое представление обратно в действие."""
        action = {}
        for key, value in hashable_action:
            if isinstance(value, tuple):
                # Преобразуем строковые представления карт обратно в объекты Card
                action[key] = [Card(card[0], card[1]) for card in value]
            else:
                action[key] = value
        return action
class CFRAgent:
    def __init__(self, iterations: int = 1000, stop_threshold: float = 0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.historical_data = defaultdict(list)
        self.regrets = defaultdict(list)

    def cfr(self, game_state: GameState, p0: float, p1: float, timeout_event: Event, result: dict) -> float:
        """Реализует алгоритм CFR для поиска оптимальной стратегии."""
        if timeout_event.is_set():
            return 0

        if game_state.is_terminal():
            return game_state.calculate_score()

        player = game_state.get_current_player()
        info_set = game_state.get_information_set()

        # Создаем новый узел, если его еще нет
        if info_set not in self.nodes:
            actions = game_state.get_actions()
            if not actions:
                return 0
            self.nodes[info_set] = CFRNode(actions)

        node = self.nodes[info_set]
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = defaultdict(float)
        node_util = 0

        # Рекурсивно вычисляем полезность для каждого действия
        for action in node.actions:
            if timeout_event.is_set():
                return 0

            next_state = game_state.apply_action(action)
            if player == 0:
                util[action] = -self.cfr(next_state, p0 * strategy[action], p1, timeout_event, result)
            else:
                util[action] = -self.cfr(next_state, p0, p1 * strategy[action], timeout_event, result)
            node_util += strategy[action] * util[action]

        # Обновляем сожаления
        for action in node.actions:
            regret = util[action] - node_util
            if player == 0:
                node.regret_sum[node._action_to_hashable(action)] += p1 * regret
            else:
                node.regret_sum[node._action_to_hashable(action)] += p0 * regret

            # Сохраняем историю сожалений
            self.regrets[info_set].append((action, regret))

        return node_util

    def train(self, timeout_event: Event, result: dict) -> None:
        """Обучает агента на заданном количестве итераций."""
        for i in range(self.iterations):
            if timeout_event.is_set():
                print(f"Training interrupted after {i} iterations due to timeout.")
                break

            # Создаем начальное состояние
            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)
            game_state = GameState(selected_cards=all_cards[:5])

            # Запускаем CFR
            self.cfr(game_state, 1, 1, timeout_event, result)

            # Проверяем сходимость каждые 100 итераций
            if i % 100 == 0:
                if self.check_convergence():
                    print(f"Training converged after {i} iterations")
                    break

        # Анализируем результаты обучения
        self.analyze_training_results()

    def get_move(self, game_state: GameState, num_cards: int, timeout_event: Event, result: dict) -> None:
        """Выбирает лучший ход на основе текущего состояния игры."""
        info_set = game_state.get_information_set()
        
        if info_set in self.nodes:
            strategy = self.nodes[info_set].get_average_strategy()
            
            # Оцениваем каждое действие
            action_values = {}
            for action in strategy:
                if timeout_event.is_set():
                    break
                value = strategy[action] * self.evaluate_action(game_state, action, timeout_event)
                action_values[action] = value

            if action_values:
                best_action = max(action_values.items(), key=lambda x: x[1])[0]
                result['move'] = best_action
                
                # Сохраняем информацию о ходе
                self.historical_data[info_set].append({
                    'action': best_action,
                    'value': action_values[best_action],
                    'state': game_state.get_information_set()
                })
            else:
                result['move'] = {'error': 'Не удалось найти подходящий ход'}
        else:
            # Если состояние не найдено, используем случайный ход
            actions = game_state.get_actions()
            if actions:
                result['move'] = random.choice(actions)
            else:
                result['move'] = {'error': 'Нет доступных ходов'}

    def evaluate_action(self, state: GameState, action: Dict, timeout_event: Event) -> float:
        """Оценивает качество действия с учетом различных факторов."""
        if timeout_event.is_set():
            return 0

        next_state = state.apply_action(action)
        
        # Базовая оценка
        score = next_state.calculate_score()
        
        # Учитываем потенциал для фантазии
        fantasy_potential = next_state._calculate_fantasy_bonus()
        
        # Учитываем риск фола
        foul_risk = next_state._calculate_foul_risk()
        
        # Учитываем количество аутов
        outs = next_state.calculate_outs()
        outs_value = len(outs) * 0.1
        
        # Учитываем исторические данные
        historical_value = self._get_historical_value(next_state.get_information_set())
        
        # Комбинируем все факторы
        final_score = (
            score +
            fantasy_potential * 2.0 +  # Высокий вес для фантазии
            outs_value -
            foul_risk * 5.0 +  # Высокий штраф за риск фола
            historical_value
        )
        
        return final_score

    def _get_historical_value(self, info_set: str) -> float:
        """Возвращает историческую ценность состояния на основе предыдущих игр."""
        if not self.historical_data[info_set]:
            return 0.0
            
        values = [data['value'] for data in self.historical_data[info_set]]
        return sum(values) / len(values)

    def analyze_training_results(self) -> None:
        """Анализирует результаты обучения для улучшения стратегии."""
        for info_set, history in self.historical_data.items():
            if not history:
                continue
                
            # Анализируем успешность различных действий
            action_success = defaultdict(list)
            for data in history:
                action_success[str(data['action'])].append(data['value'])
            
            # Вычисляем средний успех для каждого действия
            avg_success = {
                action: sum(values) / len(values)
                for action, values in action_success.items()
            }
            
            # Обновляем стратегию на основе анализа
            if info_set in self.nodes:
                node = self.nodes[info_set]
                strategy = node.get_average_strategy()
                
                # Корректируем стратегию с учетом исторического успеха
                total_success = sum(avg_success.values())
                if total_success > 0:
                    for action in strategy:
                        action_str = str(action)
                        if action_str in avg_success:
                            strategy[action] = avg_success[action_str] / total_success

    def check_convergence(self) -> bool:
        """Проверяет сходимость стратегии."""
        for node in self.nodes.values():
            strategy = node.get_average_strategy()
            uniform_strategy = 1.0 / len(node.actions)
            
            for action_prob in strategy.values():
                if abs(action_prob - uniform_strategy) > self.stop_threshold:
                    return False
        return True

# Создаем экземпляр агента
cfr_agent = CFRAgent()

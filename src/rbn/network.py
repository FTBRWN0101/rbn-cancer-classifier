import numpy as np
from typing import Callable, List, Optional, Literal, Dict, Any
RuleFn = Callable[[np.ndarray], int]

class RandomBooleanNetwork:
    """
    Random Boolean Network (RBN)

    - n_nodes = number of nodes (genes)
    - k = inputs per node
    - rule_type picks rule generator if rules not passed in
    - random_state seeds numpy for repeatable runs (optional)
    """
    def __init__(
        self,
        n_nodes: int,
        k: int,
        rules: Optional[List[RuleFn]] = None,
        rule_type: Literal["majority", "threshold", "random", "canalyzing"] = "majority",
        random_state: Optional[int] = None,):
        self.n_nodes = int(n_nodes)
        self.k = int(k)
        self.rule_type = rule_type
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.connections = self._make_connections()
        if rules is None:
            self.rules = self._make_rules()
        else:
            self.rules = rules

    def _make_connections(self) -> np.ndarray:
        """
        random input connections for each node
        returns (n_nodes, k) int array"""
        conns = np.zeros((self.n_nodes, self.k), dtype=int)
        for i in range(self.n_nodes):
            pool = list(range(self.n_nodes))
            # avoid self-loop if there are enough nodes to do it
            if len(pool) > self.k and i in pool:
                pool.remove(i)
            picks = np.random.choice(pool, size=self.k, replace=False)
            conns[i] = picks
        return conns

    def _make_rules(self) -> List[RuleFn]:
        m = {"majority": self._rules_majority,"threshold": self._rules_threshold,"canalyzing": self._rules_canalyzing,"random": self._rules_random,}
        if self.rule_type not in m:
            raise ValueError(f"Unknown rule_type: {self.rule_type}")
        return m[self.rule_type]()

    @staticmethod
    def _majority(inputs: np.ndarray) -> int:
        return int(np.sum(inputs) > (len(inputs) / 2))

    def _rules_majority(self) -> List[RuleFn]:
        return [self._majority for _ in range(self.n_nodes)]

    def _rules_threshold(self) -> List[RuleFn]:
        thresholds = np.random.randint(1, self.k + 1, size=self.n_nodes)
        out: List[RuleFn] = []
        for thr in thresholds:
            def _thr_rule(inputs: np.ndarray, t=thr) -> int:
                return int(np.sum(inputs) >= t)
            out.append(_thr_rule)
        return out

    def _rules_canalyzing(self) -> List[RuleFn]:
        out: List[RuleFn] = []
        for _ in range(self.n_nodes):
            can_idx = np.random.randint(0, self.k)
            can_val = np.random.randint(0, 2)
            out_val = np.random.randint(0, 2)
            def _can_rule(
                inputs: np.ndarray,
                idx=can_idx,
                v=can_val,
                o=out_val,
            ) -> int:
                if inputs[idx] == v:
                    return o
                return self._majority(inputs)
            out.append(_can_rule)
        return out

    def _rules_random(self) -> List[RuleFn]:
        def _rand_rule(inputs: np.ndarray) -> int:
            return np.random.randint(0, 2)
        return [_rand_rule for _ in range(self.n_nodes)]

    def step(self, state: np.ndarray) -> np.ndarray:
        """
        one synchronous update step, state is (n_nodes,) binary
        """
        nxt = np.zeros_like(state, dtype=np.int8)
        for i in range(self.n_nodes):
            ins = state[self.connections[i]]
            nxt[i] = self.rules[i](ins)
        return nxt

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "k": self.k,
            "rule_type": self.rule_type,
            "connections": self.connections,
        }

class DataDrivenRBN(RandomBooleanNetwork):
    """
    RBN with learned weighted-threshold rules. Subclass of class "RandomBooleanNetwork"
    - random connectivity stays
    - rules learned from training X using correlations + bias"""

    def __init__(self, n_nodes: int, k: int, random_state: Optional[int] = None):
        super().__init__(
            n_nodes=n_nodes,
            k=k,
            rules=None,
            rule_type="threshold",  # placeholder, overwritten in fit
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        learn weights + bias from training data
        weights[i, j] = corr(X[:, input_idx], X[:, i])
        bias[i] = mean(X[:, i]) - 0.5
        """
        print(f"  Learning Boolean rules from {len(X)} samples")
        self.weights = np.zeros((self.n_nodes, self.k))
        self.biases = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            ins = self.connections[i]
            for j, in_idx in enumerate(ins):
                c = np.corrcoef(X[:, in_idx], X[:, i])[0, 1]
                if np.isnan(c):
                    c = 0.0
                self.weights[i, j] = c
            self.biases[i] = X[:, i].mean() - 0.5
        self.rules = [self._make_learned_rule(i) for i in range(self.n_nodes)]
        self._fitted = True
        print(f"  ✓ Learned rules for {self.n_nodes} nodes")
        print(f"    Weight range: [{self.weights.min():.3f}, {self.weights.max():.3f}]")
        print(f"    Bias range: [{self.biases.min():.3f}, {self.biases.max():.3f}]")
        return self

    def _make_learned_rule(self, node_idx: int) -> RuleFn:
        def _rule(inputs: np.ndarray, idx=node_idx) -> int:
            s = float(np.dot(inputs, self.weights[idx])) + float(self.biases[idx])
            return int(s > 0)
        return _rule

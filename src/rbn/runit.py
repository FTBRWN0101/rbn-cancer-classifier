# src/rbn/runit.py
import numpy as np
from .network import RandomBooleanNetwork, DataDrivenRBN
from .simulator import RBNSimulator
from .attractor import AttractorAnalyzer
from .features import AttractorFeatureExtractor, CombinedFeatureExtractor


def main():
    """Test RBN components in isolation."""
    print("Testing RBN Components")
    print("\n1. Testing RandomBooleanNetwork with majority rule...")
    n_nodes = 30
    k = 3
    
    network = RandomBooleanNetwork(
        n_nodes=n_nodes,
        k=k,
        rules=None,
        rule_type='majority',
        random_state=42
    )
    print(f"   Network: N={n_nodes}, K={k}")
    print(f"   Connections shape: {network.connections.shape}")
    
    #2: simulate dynamics
    print("\n2. Testing RBN simulation")
    initial_state = np.random.randint(0, 2, size=n_nodes)
    simulator = RBNSimulator(network, max_steps=50)
    trajectory = simulator.run(initial_state)
    print(f"Initial state: {initial_state}")
    print(f"Trajectory length: {len(trajectory)}")
    
    #3: find attractor
    print("\n3. Testing attractor analysis")
    attractor_analyzer = AttractorAnalyzer()
    attractor = attractor_analyzer.find_attractor(trajectory)
    if attractor:
        print(f"   Attractor found!")
        print(f"   Attractor length: {len(attractor)}")
        print(f"   Attractor type: {'Point' if len(attractor) == 1 else 'Cycle'}")
    else:
        print(f"   No attractor found in trajectory")
    
    #4: extract features
    print("\n4. Testing feature extraction...")
    feature_extractor = AttractorFeatureExtractor()
    features = feature_extractor.extract(attractor)
    print(f"   Features extracted: {features}")
    print(f"   Feature names: {feature_extractor.get_feature_names()}")
    
    #5: Combined features
    print("\n5. Testing combined feature extraction")
    attractor_with_trans, attractor_start = attractor_analyzer.find_attractor_with_transient(trajectory)
    combined_extractor = CombinedFeatureExtractor()
    combined_features = combined_extractor.extract(trajectory, attractor_with_trans, attractor_start)
    print(f"   Transient length: {attractor_start}")
    print(f"   Combined features shape: {combined_features.shape}")
    print(f"   Combined features: {combined_features}")
    
    #6: Data-driven RBN (needs training data)
    print("\n6. Testing DataDrivenRBN")
    print("   Creating synthetic training data")
    n_samples = 50
    X_train = np.random.randint(0, 2, size=(n_samples, n_nodes))
    data_driven_network = DataDrivenRBN(n_nodes=n_nodes, k=k, random_state=42)
    data_driven_network.fit(X_train)
    print(f"   Data-driven network fitted successfully")
    
    #test network
    test_state = np.random.randint(0, 2, size=n_nodes)
    simulator_dd = RBNSimulator(data_driven_network, max_steps=50)
    trajectory_dd = simulator_dd.run(test_state)
    attractor_dd = attractor_analyzer.find_attractor(trajectory_dd)
    features_dd = feature_extractor.extract(attractor_dd)
    print(f"   Test trajectory length: {len(trajectory_dd)}")
    print(f"   Test attractor length: {len(attractor_dd)}")
    print(f"   Test features: {features_dd}")
    print("RBN Component Testing Complete!")
    print("\all components working. Ready for use")

if __name__ == "__main__":
    main()
# Add these imports at the top
import numpy as np
import pandas as pd
from datetime import datetime
import time
import concurrent.futures
import copy
from collections import defaultdict
import heapq
import plotly.graph_objects as go
import streamlit as st
import graphviz

# Add these new functions for the database simulation
def create_simulation_data(size=1000000):
    """Create large simulation dataset"""
    np.random.seed(42)
    data = {
        'id': range(size),
        'timestamp': pd.date_range(start='2023-01-01', periods=size, freq='S'),
        'value': np.random.normal(100, 15, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'status': np.random.choice(['active', 'inactive'], size, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

class DatabaseSimulator:
    def __init__(self, data):
        self.main_db = data
        self.replica_db = None
        self.last_sync = None
        
    def write_to_main(self, new_data):
        """Simulate write to main database"""
        with st.spinner('Writing to main database...'):
            self.main_db = pd.concat([self.main_db, new_data], ignore_index=True)
    
    def sync_replica(self):
        """Sync replica with main database"""
        with st.spinner('Syncing replica database...'):
            self.replica_db = self.main_db.copy()
            self.last_sync = datetime.now()
    
    def read_from_main(self, query_params):
        """Simulate read from main database"""
        with st.spinner('Reading from main database...'):
            return self.filter_data(self.main_db, query_params)
    
    def read_from_replica(self, query_params):
        """Read from replica database"""
        if self.replica_db is None:
            return None
        return self.filter_data(self.replica_db, query_params)
    
    @staticmethod
    def filter_data(df, query_params):
        """Filter data based on query parameters"""
        filtered = df.copy()
        for key, value in query_params.items():
            if key in df.columns:
                filtered = filtered[filtered[key] == value]
        return filtered

class OptimizedQueryProcessor:
    def __init__(self, data):
        self.data = data
        self.indexes = self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for faster querying"""
        indexes = {
            'category': defaultdict(list),
            'status': defaultdict(list)
        }
        
        for idx, row in self.data.iterrows():
            indexes['category'][row['category']].append(idx)
            indexes['status'][row['status']].append(idx)
        
        return indexes
    
    def query_data(self, query_params):
        """Query data using optimized indexes"""
        candidate_sets = []
        
        # Use indexes for filtering
        for key, value in query_params.items():
            if key in self.indexes:
                candidate_sets.append(set(self.indexes[key][value]))
        
        # Find intersection of all candidate sets
        if candidate_sets:
            result_indices = set.intersection(*candidate_sets)
            return self.data.iloc[list(result_indices)]
        
        return self.data

def run_performance_test(db_simulator, optimized_processor, query_params, num_queries=100):
    """Run performance comparison test"""
    results = {
        'main_db': [],
        'replica_db': [],
        'optimized': []
    }
    
    # Test main database
    start = datetime.now()
    for _ in range(num_queries):
        db_simulator.read_from_main(query_params)
    results['main_db'] = (datetime.now() - start).total_seconds()
    
    # Test replica database
    start = datetime.now()
    for _ in range(num_queries):
        db_simulator.read_from_replica(query_params)
    results['replica_db'] = (datetime.now() - start).total_seconds()
    
    # Test optimized query
    start = datetime.now()
    for _ in range(num_queries):
        optimized_processor.query_data(query_params)
    results['optimized'] = (datetime.now() - start).total_seconds()
    
    return results

def draw_architecture_diagrams():
    # Add to the explanation section in add_database_simulation_tab()
    st.markdown("### System Architecture Comparison")
    
    # Replica DB Approach
    st.markdown("#### 1. Replica Database Architecture")
    
    replica_diagram = """
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=lightblue];
        
        subgraph cluster_0 {
            label="Main Database Node";
            style=filled;
            color=lightgrey;
            MainDB [label="Main Database\n(Write Operations)"];
        }
        
        subgraph cluster_1 {
            label="Replica Node";
            style=filled;
            color=lightgrey;
            ReplicaDB [label="Replica Database\n(Read Operations)"];
        }
        
        Client1 [label="Client\n(Write Request)"];
        Client2 [label="Client\n(Read Request)"];
        
        Client1 -> MainDB [label="1. Write"];
        MainDB -> ReplicaDB [label="2. Sync"];
        Client2 -> ReplicaDB [label="3. Read"];
        
        {rank=same; Client1; Client2}
    }
    """
    
    # Optimized Query Approach
    st.markdown("#### 2. Optimized Query Architecture")
    
    optimized_diagram = """
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=lightblue];
        
        subgraph cluster_0 {
            label="Database Node";
            style=filled;
            color=lightgrey;
            
            MainDB [label="Main Database"];
            IndexStructure [label="Index Structures\n(Hash Tables, B-Trees)"];
            QueryOptimizer [label="Query Optimizer"];
            
            MainDB -> IndexStructure [dir=both, label="Update"];
            IndexStructure -> QueryOptimizer [label="Use"];
        }
        
        Client1 [label="Client\n(Write Request)"];
        Client2 [label="Client\n(Read Request)"];
        
        Client1 -> MainDB [label="1. Write"];
        Client2 -> QueryOptimizer [label="2. Optimized\nRead"];
        
        {rank=same; Client1; Client2}
    }
    """
    
    # Add diagrams to the UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.graphviz_chart(replica_diagram)
        st.markdown("""
        **Replica DB Approach Flow:**
        1. Writes go to main database
        2. Periodic sync to replica
        3. Reads served from replica
        
        **Key Characteristics:**
        - Write latency: Medium
        - Read latency: Low
        - Data freshness: Slightly delayed
        - Resource usage: High (2x storage)
        """)
    
    with col2:
        st.graphviz_chart(optimized_diagram)
        st.markdown("""
        **Optimized Query Flow:**
        1. Writes update main DB & indexes
        2. Reads use optimized paths
        
        **Key Characteristics:**
        - Write latency: Medium-High
        - Read latency: Very Low
        - Data freshness: Real-time
        - Resource usage: Medium
        """)
    
    # Add performance comparison table
    st.markdown("### Approach Comparison Matrix")
    comparison_data = {
        "Aspect": [
            "Write Performance",
            "Read Performance",
            "Data Consistency",
            "Resource Usage",
            "Implementation Complexity",
            "Maintenance Overhead",
            "Scalability",
            "Fault Tolerance"
        ],
        "Replica DB": [
            "⭐⭐⭐⭐ (Fast direct writes)",
            "⭐⭐⭐⭐ (No contention with writes)",
            "⭐⭐⭐ (Eventual consistency)",
            "⭐⭐ (Requires double storage)",
            "⭐⭐⭐ (Moderate)",
            "⭐⭐⭐ (Sync management needed)",
            "⭐⭐⭐⭐ (Can add more replicas)",
            "⭐⭐⭐⭐ (Built-in redundancy)"
        ],
        "Optimized Query": [
            "⭐⭐⭐ (Index updates needed)",
            "⭐⭐⭐⭐⭐ (Optimized paths)",
            "⭐⭐⭐⭐⭐ (Always current)",
            "⭐⭐⭐⭐ (Index overhead only)",
            "⭐⭐ (Complex optimization)",
            "⭐⭐⭐⭐ (Automatic index updates)",
            "⭐⭐⭐ (Limited by single node)",
            "⭐⭐ (Single point of failure)"
        ]
    }
    
    st.table(pd.DataFrame(comparison_data))



def add_database_simulation_tab():
    st.header("Database Performance Simulation")
    
    # Add explanation section
    with st.expander("📚 About This Simulation", expanded=True):
        st.markdown("""
        ### Database Query Optimization Approaches
        
        This simulation compares two different approaches to handling high-read, slow-write database scenarios:
        
        #### 1. Replica Database Approach
        - **What it is**: A copy of the main database that's updated periodically
        - **How it works**:
            * Main database handles all write operations
            * Replica database is synced at regular intervals
            * Read queries are directed to the replica
        - **Advantages**:
            * Reduces load on main database
            * Provides read availability during main DB maintenance
            * Supports geographical distribution
        - **Disadvantages**:
            * Data might be slightly outdated
            * Requires additional storage
            * Needs sync management
        
        #### 2. Optimized Query Approach
        - **What it is**: Uses specialized data structures and algorithms to speed up queries
        - **How it works**:
            * Builds indexes for frequently queried fields
            * Uses hash tables for O(1) lookups
            * Implements set operations for filtering
        - **Advantages**:
            * Always has current data
            * Uses less storage
            * No sync needed
        - **Disadvantages**:
            * Index maintenance overhead
            * Memory intensive
            * Complex implementation
        
        ### Filter Categories Explanation
        
        #### Category Filter (A, B, C, D)
        - Simulates different product categories or classifications
        - **A**: Premium products
        - **B**: Standard products
        - **C**: Economy products
        - **D**: Special items
        
        #### Status Filter (active/inactive)
        - **Active**: Currently available/in-use items (80% of data)
        - **Inactive**: Discontinued or unavailable items (20% of data)
        
        ### Performance Metrics
        - **Main DB Time**: Time taken to query the primary database
        - **Replica DB Time**: Time taken to query the replica database
        - **Optimized Query Time**: Time taken using indexed approach
        """)
    # Add the architecture diagrams
    draw_architecture_diagrams()

    # Simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        data_size = st.number_input(
            "Dataset Size", 
            min_value=1000, 
            max_value=1000000, 
            value=100000, 
            step=1000,
            help="Number of records to simulate. Larger datasets show more pronounced performance differences."
        )
        num_queries = st.number_input(
            "Number of Queries", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Number of consecutive queries to run. Higher numbers provide more stable averages."
        )
    
    with col2:
        category_filter = st.selectbox(
            "Category Filter", 
            ['A', 'B', 'C', 'D'],
            help="Select a product category to filter by. Distribution is uniform across categories."
        )
        status_filter = st.selectbox(
            "Status Filter", 
            ['active', 'inactive'],
            help="Select status to filter by. Active items are 80% of the dataset."
        )
    
    # Add technical details expander
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        ### Implementation Details
        
        #### Optimized Query Implementation
        ```python
        def query_data(self, query_params):
            candidate_sets = []
            for key, value in query_params.items():
                if key in self.indexes:
                    candidate_sets.append(set(self.indexes[key][value]))
            if candidate_sets:
                result_indices = set.intersection(*candidate_sets)
                return self.data.iloc[list(result_indices)]
            return self.data
        ```
        
        #### Replica DB Implementation
        ```python
        def sync_replica(self):
            time.sleep(0.5)  # Simulate sync delay
            self.replica_db = self.main_db.copy()
            self.last_sync = datetime.now()
        ```
        
        - Index Build Time Complexity: O(n)
        - Query Time Complexity: O(m) where m is the size of the smallest index set
        - Space Complexity: O(n) for indexes
        """)
    
    if st.button("Run Simulation", key="run_sim"):
        with st.spinner("Running simulation..."):
            # [Rest of the simulation code remains the same]
            data = create_simulation_data(data_size)
            db_simulator = DatabaseSimulator(data)
            optimized_processor = OptimizedQueryProcessor(data)
            db_simulator.sync_replica()
            
            query_params = {
                'category': category_filter,
                'status': status_filter
            }
            
            results = run_performance_test(
                db_simulator, 
                optimized_processor, 
                query_params, 
                num_queries
            )
            
            # [Display results code remains the same]
            st.subheader("Performance Results")
            
            fig = go.Figure(data=[
                go.Bar(name='Main DB', x=['Query Time'], y=[results['main_db']]),
                go.Bar(name='Replica DB', x=['Query Time'], y=[results['replica_db']]),
                go.Bar(name='Optimized Query', x=['Query Time'], y=[results['optimized']])
            ])
            
            fig.update_layout(
                title="Query Performance Comparison",
                yaxis_title="Time (seconds)",
                barmode='group'
            )
            
            st.plotly_chart(fig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Main DB Time", f"{results['main_db']:.3f}s")
            with col2:
                st.metric("Replica DB Time", f"{results['replica_db']:.3f}s")
            with col3:
                st.metric("Optimized Query Time", f"{results['optimized']:.3f}s")
            
            fastest = min(results.items(), key=lambda x: x[1])
            speedup = results['main_db'] / results[fastest[0]]
            
            st.subheader("Performance Analysis")
            st.write(f"The fastest approach was **{fastest[0]}** with a **{speedup:.2f}x** speedup over the main database.")
            
            st.subheader("Recommendations")
            if results['replica_db'] < results['optimized']:
                st.write("The replica database approach performs better for this workload. Consider:")
                st.write("- Implementing a regular sync schedule")
                st.write("- Monitoring sync delays")
                st.write("- Setting up failover mechanisms")
            else:
                st.write("The optimized query approach performs better for this workload. Consider:")
                st.write("- Implementing more sophisticated indexing")
                st.write("- Adding caching mechanisms")
                st.write("- Optimizing query patterns")

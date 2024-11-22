import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import pygwalker as pyg
import requests
from community import community_louvain
from transformers.pipelines import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy import sparse
import os
import docker
from kubernetes import client, config

# Access the GitHub token from secrets
token = st.secrets["GITHUB_TOKEN"]

def fetch_github_data(repo_name, date_range, token):
    headers = {'Authorization': f'token {token}'}
    base_url = f'https://api.github.com/repos/{repo_name}'

    # Fetch commits with pagination
    commits_url = f'{base_url}/commits'
    all_commits = []
    page = 1
    while True:
        params = {
            'since': date_range[0].isoformat(),
            'until': date_range[1].isoformat(),
            'per_page': 100,
            'page': page
        }
        commits_response = requests.get(commits_url, headers=headers, params=params)
        commits_data = commits_response.json()
        if not commits_data or 'message' in commits_data:
            break
        all_commits.extend(commits_data)
        page += 1

    # Process commits into detailed data
    commit_details = []
    for commit in all_commits:
        commit_info = {
            'date': commit['commit']['author']['date'][:10],
            'author': commit['commit']['author']['name'],
            'message': commit['commit']['message'],
            'files_changed': len(commit.get('files', []))
        }
        commit_details.append(commit_info)

    # Convert to DataFrame for analysis
    commit_df = pd.DataFrame(commit_details)

    # Fetch contributors
    contributors_url = f'{base_url}/contributors'
    contributors_response = requests.get(contributors_url, headers=headers)
    contributors_data = contributors_response.json()
    contributors = [contributor['login'] for contributor in contributors_data]
    contributions = [contributor['contributions'] for contributor in contributors_data]

    # Fetch code changes (using commits as a proxy)
    code_changes = np.random.randint(100, 1000, size=3)  # Placeholder for actual logic

    return commit_df, contributors, contributions, code_changes

def fetch_github_repo_stats(repo_name, token):
    headers = {'Authorization': f'token {token}'}
    base_url = f'https://api.github.com/repos/{repo_name}'
    
    # Fetch repository statistics
    repo_response = requests.get(base_url, headers=headers)
    repo_data = repo_response.json()
    
    stars = repo_data.get('stargazers_count', 0)
    forks = repo_data.get('forks_count', 0)
    open_issues = repo_data.get('open_issues_count', 0)
    
    return stars, forks, open_issues

def fetch_github_actions_status(repo, token):
    """Fetch GitHub Actions workflow runs"""
    try:
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        url = f'https://api.github.com/repos/{repo}/actions/runs'
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            runs = response.json()['workflow_runs']
            run_details = []
            
            for run in runs[:20]:  # Get latest 20 runs
                run_details.append({
                    'workflow': run['name'],
                    'status': run['status'],
                    'conclusion': run['conclusion'],
                    'branch': run['head_branch'],
                    'created_at': run['created_at'],
                    'updated_at': run['updated_at']
                })
            
            return run_details
        else:
            st.error(f"Error fetching workflow runs: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error accessing GitHub API: {str(e)}")
        return None


def fetch_additional_insights(repo_name, token):
    headers = {'Authorization': f'token {token}'}
    base_url = f'https://api.github.com/repos/{repo_name}'

    # Fetch pull requests
    prs_url = f'{base_url}/pulls'
    prs_response = requests.get(prs_url, headers=headers, params={'state': 'all'})
    prs_data = prs_response.json()
    num_merged_prs = sum(1 for pr in prs_data if pr['merged_at'] is not None)
    avg_time_to_merge = np.mean([(datetime.strptime(pr['merged_at'], '%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days for pr in prs_data if pr['merged_at']])

    # Fetch issues
    issues_url = f'{base_url}/issues'
    issues_response = requests.get(issues_url, headers=headers, params={'state': 'all'})
    issues_data = issues_response.json()
    num_open_issues = sum(1 for issue in issues_data if issue['state'] == 'open')
    num_closed_issues = sum(1 for issue in issues_data if issue['state'] == 'closed')

    # Fetch stars and forks
    repo_response = requests.get(base_url, headers=headers)
    repo_data = repo_response.json()
    stars = repo_data['stargazers_count']
    forks = repo_data['forks_count']

        # Fetch contributors
    contributors_url = f'{base_url}/contributors'
    contributors_response = requests.get(contributors_url, headers=headers)
    contributors_data = contributors_response.json()

    return prs_data, issues_data, num_merged_prs, avg_time_to_merge, num_open_issues, num_closed_issues, stars, forks, contributors_data


def process_query(query, prs_data, issues_data, contributors_data):
    # Simple NLP model to interpret queries
    nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["pull requests to review", "stale issues", "contributors"]

    result = nlp(query, labels)
    label = result['labels'][0]

    if label == "pull requests to review":
        unreviewed_prs = [pr for pr in prs_data if pr.get('comments', 0) == 0]
        return unreviewed_prs
    elif label == "stale issues":
        stale_issues = [issue for issue in issues_data if issue['state'] == 'open' and (datetime.now() - datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days > 30]
        return stale_issues
    elif label == "contributors":
        return contributors_data
    else:
        return []


def create_comprehensive_dataset(commit_df, contributors, prs_data, issues_data):
    # Create a DataFrame for contributors
    contributors_df = pd.DataFrame({
        'Contributor': contributors,
        'Total Commits': [commit_df[commit_df['author'] == contributor].shape[0] for contributor in contributors]
    })

    # Create a DataFrame for pull requests
    prs_df = pd.DataFrame(prs_data)
    prs_df['Days Open'] = (pd.to_datetime(prs_df['closed_at']) - pd.to_datetime(prs_df['created_at'])).dt.days
    prs_df['Type'] = 'PR'

    # Create a DataFrame for issues
    issues_df = pd.DataFrame(issues_data)
    issues_df['Days Open'] = (pd.to_datetime(issues_df['closed_at']) - pd.to_datetime(issues_df['created_at'])).dt.days
    issues_df['Type'] = 'Issue'

    # Combine all data into a comprehensive DataFrame
    comprehensive_df = pd.concat([
        contributors_df,
        prs_df[['Type', 'state', 'created_at', 'closed_at', 'Days Open']],
        issues_df[['Type', 'state', 'created_at', 'closed_at', 'Days Open']]
    ], axis=0, ignore_index=True)

    return comprehensive_df


def provide_recommendations(prs_data, issues_data):
    # Identify PRs not reviewed
    unreviewed_prs = [pr for pr in prs_data if pr.get('comments', 0) == 0]
    stale_issues = [issue for issue in issues_data if issue['state'] == 'open' and (datetime.now() - datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days > 30]
    
    recommendations = []
    if unreviewed_prs:
        recommendations.append(f"There are {len(unreviewed_prs)} PRs that need review.")
    if stale_issues:
        recommendations.append(f"There are {len(stale_issues)} stale issues that need attention.")
    
    return recommendations, unreviewed_prs, stale_issues

def create_real_interaction_network(contributors, commit_df):
    # Create a graph based on co-authored commits
    G = nx.Graph()
    
    # Add nodes for each contributor
    for contributor in contributors:
        G.add_node(contributor)
    
    # Add edges based on co-authored commits
    for _, row in commit_df.iterrows():
        authors = row['author'].split(', ')
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                if G.has_edge(authors[i], authors[j]):
                    G[authors[i]][authors[j]]['weight'] += 1
                else:
                    G.add_edge(authors[i], authors[j], weight=1)
    
    # Plot the network graph using Plotly
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_size = []
    node_color = []  # Initialize node_color as a list
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.degree[node] * 10)  # Scale size by degree
        node_color.append(G.degree[node])  # Append color based on degree

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,  # Use the node_color list
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Contributors Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Contributor interactions",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def create_example_network(contributors, contributions):
    # Create a random network graph
    G = nx.Graph()
    
    # Normalize contributions to a range of 10 to 20
    min_size, max_size = 10, 20
    min_contribution = min(contributions)
    max_contribution = max(contributions)
    node_sizes = [
        min_size + (max_size - min_size) * (contribution - min_contribution) / (max_contribution - min_contribution)
        for contribution in contributions
    ]
    
    # Add nodes with sizes
    for contributor, size in zip(contributors, node_sizes):
        G.add_node(contributor, size=size)
    
    # Add random edges
    for i in range(len(contributors)):
        for j in range(i + 1, len(contributors)):
            if np.random.rand() > 0.7:  # Randomly decide to add an edge
                G.add_edge(contributors[i], contributors[j])
    
    # Plot the network graph using Plotly
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]['size'])
        node_color.append(G.degree[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,
            size=node_size,  # Use the normalized sizes
            colorbar=dict(
                thickness=15,
                title='Contributions',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Example Network with Random Connections',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Example of how it would look like if contributors interacted - Co-authored commits",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def display_chaoss_metrics(commit_df, num_open_issues, num_closed_issues):
    # Example CHAOSS metric: Issue closure rate
    if num_open_issues + num_closed_issues > 0:
        issue_closure_rate = num_closed_issues / (num_open_issues + num_closed_issues)
    else:
        issue_closure_rate = 0
    st.metric("Issue Closure Rate", f"{issue_closure_rate:.2%}")

def detect_communities(G):
    # Use the Louvain method to detect communities
    partition = community_louvain.best_partition(G)
    return partition

def detect_communities(G):
    # Ensure G is a NetworkX graph
    if isinstance(G, nx.Graph):
        # Use the Louvain method to detect communities
        partition = community_louvain.best_partition(G)
        return partition
    else:
        raise ValueError("Input is not a NetworkX graph")

def prepare_issue_data(issues_data):
    """Prepare issue data for ML model"""
    issues_df = pd.DataFrame([{
        'title': issue['title'],
        'body': issue.get('body', ''),
        'created_at': datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ'),
        'state': issue['state'],
        'comments': issue['comments'],
        'labels': [label['name'] for label in issue.get('labels', [])],
        'is_stale': (datetime.now() - datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days > 30
    } for issue in issues_data])
    
    return issues_df

def train_issue_model(issues_df):
    """Train ML model for issue classification"""
    # Prepare features
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(issues_df['title'] + ' ' + issues_df['body'].fillna(''))
    
    # Additional features - convert to numeric types explicitly
    X_meta = sparse.csr_matrix(np.array([
        issues_df['comments'].astype(float),
        issues_df['labels'].apply(len).astype(float) > 0
    ]).T)
    
    # Combine features using scipy's hstack
    X = sparse.hstack([X_text, X_meta])
    y = issues_df['is_stale']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    return model, vectorizer

def predict_issue_status(model, vectorizer, title, body, comments, has_labels):
    """Predict if an issue is likely to become stale"""
    text = f"{title} {body}"
    X_text = vectorizer.transform([text])
    X_meta = sparse.csr_matrix([[float(comments), float(has_labels)]])
    
    X = sparse.hstack([X_text, X_meta])
    
    probability = model.predict_proba(X)[0][1]
    return probability


def get_augur_docker_metrics():
    """Get metrics specifically for augurlabs/augur-docker containers"""
    try:
        import docker
        client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
        
        # List all containers and filter for augurlabs/augur-docker containers
        augur_containers = client.containers.list(
            all=True,
            filters={
                'name': ['augur', 'chaoss'],  # Container names
                'ancestor': [
                    'augurlabs/augur-docker',  # Main image
                    'augurlabs/augur',         # Alternative image name
                    'postgrest/postgrest',     # Related services
                    'postgres',
                    'redis'
                ]
            }
        )
        
        if not augur_containers:
            st.warning("No augurlabs/augur-docker containers found.")
            
            # Display help for setting up augurlabs/augur-docker
            st.info("""
            To run augurlabs/augur-docker:
            1. Clone the repository:
               ```bash
               git clone https://github.com/augurlabs/augur-docker.git
               cd augur-docker
               ```
            2. Build and run:
               ```bash
               docker-compose up -d
               ```
            
            This will start:
            - Augur API server
            - PostgreSQL database
            - Redis cache
            - PostgREST API
            """)
            return None
            
        container_metrics = []
        
        for container in augur_containers:
            try:
                metrics = {
                    'container_id': container.short_id,
                    'name': container.name,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'no-tag',
                    'created': container.attrs['Created'],
                    'service_type': get_augur_service_type(container.name)
                }
                
                # Only get stats if container is running
                if container.status == 'running':
                    stats = container.stats(stream=False)
                    
                    # CPU metrics
                    if 'cpu_stats' in stats and 'cpu_usage' in stats['cpu_stats']:
                        metrics['cpu_usage'] = stats['cpu_stats']['cpu_usage']['total_usage']
                        metrics['cpu_percent'] = calculate_cpu_percent(stats)
                    
                    # Memory metrics
                    if 'memory_stats' in stats:
                        metrics['memory_usage'] = stats['memory_stats'].get('usage', 0)
                        metrics['memory_limit'] = stats['memory_stats'].get('limit', 0)
                        metrics['memory_percent'] = (metrics['memory_usage'] / metrics['memory_limit'] * 100) if metrics['memory_limit'] > 0 else 0
                    
                    # Network metrics
                    if 'networks' in stats and 'eth0' in stats['networks']:
                        metrics['network_rx'] = stats['networks']['eth0']['rx_bytes']
                        metrics['network_tx'] = stats['networks']['eth0']['tx_bytes']
                else:
                    # Add placeholder metrics for non-running containers
                    metrics.update({
                        'cpu_usage': 0,
                        'cpu_percent': 0,
                        'memory_usage': 0,
                        'memory_limit': 0,
                        'memory_percent': 0,
                        'network_rx': 0,
                        'network_tx': 0
                    })
                
                container_metrics.append(metrics)
                
            except Exception as container_error:
                st.warning(f"Error getting metrics for container {container.name}: {str(container_error)}")
                continue
        
        if container_metrics:
            df = pd.DataFrame(container_metrics)
            
            # Convert bytes to MB for memory
            df['memory_usage_mb'] = df['memory_usage'] / (1024 * 1024)
            df['memory_limit_mb'] = df['memory_limit'] / (1024 * 1024)
            
            # Convert bytes to MB for network
            df['network_rx_mb'] = df['network_rx'] / (1024 * 1024)
            df['network_tx_mb'] = df['network_tx'] / (1024 * 1024)
            
            return df
        else:
            st.info("No container metrics available")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to Docker: {str(e)}")
        if "Permission denied" in str(e):
            st.error("Permission denied. Make sure you have Docker permissions:")
            st.code("sudo usermod -aG docker $USER")
            st.error("Then log out and log back in.")
        return None


def get_augur_service_type(container_name):
    """Determine the service type from container name"""
    name_lower = container_name.lower()
    if 'api' in name_lower or 'backend' in name_lower:
        return 'API Server'
    elif 'postgres' in name_lower or 'db' in name_lower:
        return 'Database'
    elif 'redis' in name_lower:
        return 'Cache'
    elif 'postgrest' in name_lower:
        return 'PostgREST'
    elif 'worker' in name_lower:
        return 'Worker'
    else:
        return 'Other'


def calculate_cpu_percent(stats):
    """Calculate CPU percentage from Docker stats"""
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                  stats['precpu_stats']['system_cpu_usage']
    
    if system_delta > 0 and cpu_delta > 0:
        return (cpu_delta / system_delta) * 100.0
    return 0.0

def display_cicd_details():
    """Display CI/CD and Docker infrastructure details"""
    st.header("Augur Docker Infrastructure")
    
    # Create tabs with full width using columns
    col1 = st.columns(1)[0]  # Use full width
    with col1:
        docker_tab, cicd_tab = st.tabs([
            "Augur Docker Containers", "CI/CD Status"
        ])
    
    with docker_tab:
        st.subheader("Active Augur Docker Containers")
        
        try:
            import docker
            client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
            docker_info = client.info()
            st.success(f"Docker is running. Version: {docker_info['ServerVersion']}")
            
            # Add option to start Augur Docker containers
            if st.button("Start Augur Docker Environment"):
                try:
                    # Use more explicit commands and check for existing directory
                    commands = """
                        rm -rf /tmp/augur-docker
                        git clone https://github.com/augurlabs/augur-docker.git /tmp/augur-docker
                        cd /tmp/augur-docker
                        cp example.env .env
                        docker pull augurlabs/augur-augur-db-1
                        docker pull augurlabs/augur-augur-1
                        docker pull augurlabs/augur-redis-1
                        docker pull augurlabs/augur-rabbitmq-1
                        docker-compose --env-file ./environment.txt up -d --build
                    """
                    result = os.system(commands)
                    
                    if result == 0:
                        st.success("Augur Docker environment started successfully!")
                        
                        # Add verification and troubleshooting instructions
                        st.info("""
                        The containers are starting up. Please wait a few minutes for all services to initialize.
                        
                        To verify the setup:
                        
                        1. First, check if containers are running:
                        ```bash
                        docker ps | grep augur
                        ```
                        
                        2. Check container logs for startup progress:
                        ```bash
                        cd /tmp/augur-docker
                        docker-compose logs -f
                        ```
                        
                        3. Common issues and solutions:
                        - If port 5002 is already in use:
                          ```bash
                          sudo lsof -i :5002  # Check what's using port 5002
                          docker-compose down  # Stop containers
                          docker-compose up -d # Restart containers
                          ```
                        
                        - If containers fail to start:
                          ```bash
                          docker-compose down
                          docker system prune -f
                          docker-compose up -d
                          ```
                        
                        - Check individual container logs:
                          ```bash
                          docker logs augur-augur-db-1
                          docker logs augur-augur-1
                          docker logs augur-redis-1
                          docker logs augur-rabbitmq-1
                          ```
                        
                        4. Once containers are running, the services should be available at:
                        - Frontend: http://localhost:5002
                        - API: http://localhost:5002/api/unstable/
                        
                        Note: It may take several minutes for all services to fully initialize.
                        """)
                    else:
                        st.error("Failed to start Augur Docker environment. Check the logs for details.")
                        
                except Exception as e:
                    st.error(f"Error starting Augur Docker environment: {str(e)}")
                    st.error("Please try the manual setup process:")
                    st.code("""
                        git clone https://github.com/augurlabs/augur-docker.git
                        cd augur-docker
                        cp example.env .env
                        docker pull augurlabs/augur-augur-db-1
                        docker pull augurlabs/augur-augur-1
                        docker pull augurlabs/augur-redis-1
                        docker pull augurlabs/augur-rabbitmq-1
                        docker-compose --env-file ./environment.txt up -d --build
                    """)
            
            # Display only augur-docker related containers
            containers = client.containers.list(
                all=True,
                filters={
                    'name': [
                        'augur-augur-db-1',
                        'augur-augur-1',
                        'augur-redis-1',
                        'augur-rabbitmq-1'
                    ]
                }
            )

            if containers:
                st.write(f"Found {len(containers)} Augur Docker containers")
                container_info = []
                for container in containers:
                    container_info.append({
                        'Name': container.name,
                        'Status': container.status,
                        'Image': container.image.tags[0] if container.image.tags else 'no-tag',
                        'Type': get_augur_service_type(container.name)
                    })
                st.dataframe(pd.DataFrame(container_info), use_container_width=True)
            else:
                st.warning("No Augur Docker containers found. Would you like to start them?")
            
        except Exception as e:
            st.error(f"Docker is not accessible: {str(e)}")
            if "Permission denied" in str(e):
                st.error("Permission denied. Make sure you have Docker permissions:")
                st.code("sudo usermod -aG docker $USER")
                st.error("Then log out and log back in.")
        
        # Display help for setting up augurlabs/augur-docker
        with st.expander("Setup Instructions", expanded=True):
            st.markdown("""
            ### To run augurlabs/augur-docker:
            
            1. Clone the repository:
            ```bash
            git clone https://github.com/augurlabs/augur-docker.git
            cd augur-docker
            ```
            
            2. Configure environment:
            ```bash
            cp example.env .env
            # Edit .env with your settings
            ```
            
            3. Pull the required images:
            ```bash
            docker pull augurlabs/augur-augur-db-1
            docker pull augurlabs/augur-augur-1
            docker pull augurlabs/augur-redis-1
            docker pull augurlabs/augur-rabbitmq-1
            ```
            
            4. Build and run:
            ```bash
            docker-compose --env-file ./environment.txt up -d --build
            ```
            
            This will start:
            - Augur Core (augur-augur-1)
            - PostgreSQL database (augur-augur-db-1)
            - Redis cache (augur-redis-1)
            - RabbitMQ message queue (augur-rabbitmq-1)
            """)
        
        # Add refresh button with full width
        col1 = st.columns(1)[0]
        with col1:
            if st.button('Refresh Metrics', use_container_width=True):
                st.experimental_rerun()
    
    with cicd_tab:
        st.subheader("CI/CD Workflow Status")
        
        # Fetch GitHub Actions workflow runs
        #token = ''  # Make sure to set this in your Streamlit secrets
        run_details = fetch_github_actions_status("chaoss/augur", token)
        
        if run_details:
            # Convert to DataFrame
            df_runs = pd.DataFrame(run_details)
            
            # Display workflow runs with full width
            st.write("Latest 20 Workflow Runs")
            st.dataframe(df_runs, use_container_width=True)
            
            # Create visualization of workflow status
            status_counts = df_runs['status'].value_counts()
            
            # Use a single column for full width
            col_full = st.columns(1)[0]
            with col_full:
                st.plotly_chart(
                    px.pie(values=status_counts.values, 
                          names=status_counts.index,
                          title='Workflow Status Distribution'),
                    use_container_width=True
                )
            
            # Metrics in a row using columns
            col1, col2, col3 = st.columns(3)
            with col1:
                success_rate = (df_runs['status'] == 'success').mean() * 100
                st.metric("Workflow Success Rate", f"{success_rate:.1f}%")
            with col2:
                st.metric("Total Workflows", len(df_runs))
            with col3:
                st.metric("Active Workflows", len(df_runs[df_runs['status'] == 'in_progress']))
        else:
            st.warning("No workflow runs available.")
        
        # Add refresh button with full width using container
        container = st.container()
        with container:
            if st.button('Refresh Status', use_container_width=True):
                st.experimental_rerun()


def setup_kubernetes_client():
    """Initialize Kubernetes client"""
    try:
        # Try loading from default kubeconfig
        config.load_kube_config()
        return client.CoreV1Api(), client.AppsV1Api()
    except Exception as e:
        # If no config found, provide instructions
        st.error("No Kubernetes configuration found.")
        
        # Create tabs for different setup options
        setup_tab, demo_tab = st.tabs(["Setup Instructions", "Demo Mode"])
        
        with setup_tab:
            st.markdown("""
            ### Option 1: Set up with an existing Kubernetes cluster
            
            1. Install kubectl:
            ```bash
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
            ```
            
            2. Copy your cluster's kubeconfig to:
            ```bash
            mkdir -p ~/.kube
            # Copy your kubeconfig file to ~/.kube/config
            chmod 600 ~/.kube/config
            ```
            
            ### Option 2: Set up a local cluster with Minikube
            
            1. Install Minikube:
            ```bash
            curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
            sudo install minikube-linux-amd64 /usr/local/bin/minikube
            ```
            
            2. Start a cluster:
            ```bash
            minikube start
            ```
            
            ### Option 3: Set up with Kind (Kubernetes in Docker)
            
            1. Install Kind:
            ```bash
            curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
            chmod +x ./kind
            sudo mv ./kind /usr/local/bin/kind
            ```
            
            2. Create a cluster:
            ```bash
            kind create cluster --name augur-cluster
            ```
            
            ### Verify Your Setup
            
            After setting up any of the above options:
            ```bash
            kubectl cluster-info
            kubectl get nodes
            ```
            
            ### Common Issues:
            
            1. Permission denied:
            ```bash
            sudo chown $(id -u):$(id -g) $HOME/.kube/config
            ```
            
            2. Multiple contexts:
            ```bash
            kubectl config get-contexts
            kubectl config use-context <your-context>
            ```
            
            3. Invalid certificate:
            ```bash
            kubectl config view --raw
            # Check if certificates are valid and not expired
            ```
            """)
            
            if st.button("Check Connection"):
                try:
                    config.load_kube_config()
                    st.success("Successfully connected to Kubernetes cluster!")
                    return client.CoreV1Api(), client.AppsV1Api()
                except Exception as check_error:
                    st.error(f"Still unable to connect: {str(check_error)}")
        
        with demo_tab:
            st.markdown("""
            ### Demo Mode
            
            While you set up your Kubernetes cluster, you can explore the functionality 
            using demo data that simulates a typical Augur deployment.
            """)
            
            if st.button("Use Demo Data"):
                return "DEMO_MODE", "DEMO_MODE"
        
        return None, None
    


def create_deployment_visualization(resources):
    """Create the actual visualization from resources"""
    # Create node graph
    nodes = []
    edges = []
    
    # Add pods
    for pod in resources['pods']:
        nodes.append({
            'id': pod['metadata']['name'],
            'label': f"Pod: {pod['metadata']['name']}",
            'type': 'pod',
            'cpu': pod['spec']['containers'][0]['resources']['requests'].get('cpu', 'N/A'),
            'memory': pod['spec']['containers'][0]['resources']['requests'].get('memory', 'N/A')
        })
    
    # Add services and edges
    for svc in resources['services']:
        nodes.append({
            'id': svc['metadata']['name'],
            'label': f"Service: {svc['metadata']['name']}",
            'type': 'service'
        })
        
        # Connect services to pods
        for pod in resources['pods']:
            if pod['metadata'].get('labels', {}).get('app') == 'augur':
                edges.append({
                    'from': svc['metadata']['name'],
                    'to': pod['metadata']['name']
                })
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['id']],
            y=[0 if node['type'] == 'service' else 1],
            mode='markers+text',
            name=node['label'],
            text=[node['label']],
            marker=dict(
                size=30,
                color='blue' if node['type'] == 'service' else 'green'
            )
        ))
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge['from'], edge['to']],
            y=[0, 1],
            mode='lines',
            line=dict(color='gray'),
            showlegend=False
        ))
    
    fig.update_layout(
        title='Kubernetes Deployment Visualization',
        showlegend=True,
        height=600
    )
    
    return fig

def get_kubernetes_resources():
    """Fetch Kubernetes resources related to Augur/8Knot"""
    core_api, apps_api = setup_kubernetes_client()
    
    # Handle demo mode
    if core_api == "DEMO_MODE":
        return {
            'pods': [
                {
                    'metadata': {
                        'name': 'augur-frontend-pod',
                        'labels': {'app': 'augur', 'tier': 'frontend'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '200m',
                                    'memory': '256Mi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                },
                {
                    'metadata': {
                        'name': 'augur-backend-pod',
                        'labels': {'app': 'augur', 'tier': 'backend'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '512Mi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                },
                {
                    'metadata': {
                        'name': 'augur-db-pod',
                        'labels': {'app': 'augur', 'tier': 'database'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '300m',
                                    'memory': '1Gi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                }
            ],
            'services': [
                {
                    'metadata': {'name': 'augur-frontend-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'frontend'},
                        'type': 'LoadBalancer',
                        'ports': [{'port': 80, 'targetPort': 5000}]
                    }
                },
                {
                    'metadata': {'name': 'augur-backend-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'backend'},
                        'type': 'ClusterIP',
                        'ports': [{'port': 5000}]
                    }
                },
                {
                    'metadata': {'name': 'augur-db-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'database'},
                        'type': 'ClusterIP',
                        'ports': [{'port': 5432}]
                    }
                }
            ],
            'deployments': [
                {
                    'metadata': {'name': 'augur-frontend'},
                    'spec': {
                        'replicas': 2,
                        'selector': {'matchLabels': {'app': 'augur', 'tier': 'frontend'}}
                    }
                },
                {
                    'metadata': {'name': 'augur-backend'},
                    'spec': {
                        'replicas': 3,
                        'selector': {'matchLabels': {'app': 'augur', 'tier': 'backend'}}
                    }
                }
            ]
        }
    
    if not core_api or not apps_api:
        return None
    
    try:
        # Get resources with label selector for Augur
        pods = core_api.list_pod_for_all_namespaces(label_selector='app=augur')
        services = core_api.list_service_for_all_namespaces(label_selector='app=augur')
        
        return {
            'pods': pods.items,
            'services': services.items
        }
    except Exception as e:
        st.error(f"Error fetching Kubernetes resources: {str(e)}")
        return None
    
def visualize_kubernetes_deployment():
    """Create visualization of Kubernetes deployment"""
    st.subheader("Kubernetes Network Topology")
    
    # Make the demo button more prominent
    st.write("No Kubernetes resources found. Click below to see a demo deployment.")
    show_demo = st.button("Show Demo Deployment", type="primary")
    
    if show_demo:
        demo_resources = {
            'pods': [
                {
                    'metadata': {
                        'name': 'augur-frontend-pod',
                        'labels': {'app': 'augur', 'tier': 'frontend'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '200m',
                                    'memory': '256Mi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                },
                {
                    'metadata': {
                        'name': 'augur-backend-pod',
                        'labels': {'app': 'augur', 'tier': 'backend'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '500m',
                                    'memory': '512Mi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                },
                {
                    'metadata': {
                        'name': 'augur-db-pod',
                        'labels': {'app': 'augur', 'tier': 'database'}
                    },
                    'spec': {
                        'containers': [{
                            'resources': {
                                'requests': {
                                    'cpu': '300m',
                                    'memory': '1Gi'
                                }
                            }
                        }]
                    },
                    'status': {'phase': 'Running'}
                }
            ],
            'services': [
                {
                    'metadata': {'name': 'augur-frontend-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'frontend'},
                        'type': 'LoadBalancer',
                        'ports': [{'port': 80, 'targetPort': 5000}]
                    }
                },
                {
                    'metadata': {'name': 'augur-backend-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'backend'},
                        'type': 'ClusterIP',
                        'ports': [{'port': 5000}]
                    }
                },
                {
                    'metadata': {'name': 'augur-db-service'},
                    'spec': {
                        'selector': {'app': 'augur', 'tier': 'database'},
                        'type': 'ClusterIP',
                        'ports': [{'port': 5432}]
                    }
                }
            ]
        }
        
        # Display demo data
        st.success("✅ Showing demo deployment data")
        
        # Resource Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resource Overview")
            st.metric("Total Pods", len(demo_resources['pods']))
            st.metric("Total Services", len(demo_resources['services']))
            
            # Pod Details
            st.subheader("Pod Details")
            pod_data = []
            for pod in demo_resources['pods']:
                pod_data.append({
                    'Name': pod['metadata']['name'],
                    'Status': pod['status']['phase'],
                    'CPU': pod['spec']['containers'][0]['resources']['requests']['cpu'],
                    'Memory': pod['spec']['containers'][0]['resources']['requests']['memory']
                })
            st.dataframe(pd.DataFrame(pod_data))
        
        with col2:
            st.subheader("Service Details")
            service_data = []
            for svc in demo_resources['services']:
                service_data.append({
                    'Name': svc['metadata']['name'],
                    'Type': svc['spec'].get('type', 'ClusterIP'),
                    'Port': svc['spec']['ports'][0]['port']
                })
            st.dataframe(pd.DataFrame(service_data))
        
        # Network Topology
        st.subheader("Network Topology")
        fig = create_network_visualization(demo_resources)
        st.plotly_chart(fig, use_container_width=True)

def create_network_visualization(resources):
    """Create network topology visualization"""
    fig = go.Figure()
    
    # Define node positions
    service_y = 0
    pod_y = 1
    
    # Add services
    for i, svc in enumerate(resources['services']):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[service_y],
            mode='markers+text',
            name=svc['metadata']['name'],
            text=[svc['metadata']['name']],
            marker=dict(size=30, color='blue'),
            textposition="bottom center"
        ))
    
    # Add pods
    for i, pod in enumerate(resources['pods']):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[pod_y],
            mode='markers+text',
            name=pod['metadata']['name'],
            text=[pod['metadata']['name']],
            marker=dict(size=30, color='green'),
            textposition="top center"
        ))
    
    # Update layout
    fig.update_layout(
        title='Kubernetes Network Topology',
        showlegend=False,
        height=600,
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, 1.5]
        )
    )
    
    return fig

def create_augur_manifests():
    """Create Kubernetes YAML manifests for Augur deployment"""
    augur_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: augur-deployment
  labels:
    app: augur
spec:
  replicas: 1
  selector:
    matchLabels:
      app: augur
  template:
    metadata:
      labels:
        app: augur
    spec:
      containers:
      - name: augur
        image: augurlabs/augur:latest
        ports:
        - containerPort: 5000
        env:
        - name: AUGUR_DB_HOST
          value: augur-db
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: augur-service
spec:
  selector:
    app: augur
  ports:
  - port: 5000
    targetPort: 5000
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: augur-db
  labels:
    app: augur-db
spec:
  replicas: 1
  selector:
    matchLabels:
      app: augur-db
  template:
    metadata:
      labels:
        app: augur-db
    spec:
      containers:
      - name: postgres
        image: postgres:13
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: augur
        - name: POSTGRES_USER
          value: augur
        - name: POSTGRES_PASSWORD
          value: augur
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: augur-db
spec:
  selector:
    app: augur-db
  ports:
  - port: 5432
    targetPort: 5432
"""
    return augur_deployment

def deploy_augur_to_kubernetes():
    """Deploy Augur to Kubernetes cluster"""
    st.subheader("Deploy Augur to Kubernetes")
    
    # Create columns for deployment options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Deployment Configuration")
        replicas = st.number_input("Number of Replicas", min_value=1, max_value=10, value=1)
        db_storage = st.number_input("Database Storage (Gi)", min_value=1, max_value=100, value=10)
        
        # Add deployment button
        if st.button("Deploy Augur"):
            try:
                # Create manifests
                manifests = create_augur_manifests()
                
                # Write manifests to temporary file
                with open("/tmp/augur-manifests.yaml", "w") as f:
                    f.write(manifests)
                
                # Apply manifests using kubectl
                result = os.system("kubectl apply -f /tmp/augur-manifests.yaml")
                
                if result == 0:
                    st.success("Augur deployment started successfully!")
                else:
                    st.error("Failed to deploy Augur. Check the logs for details.")
            except Exception as e:
                st.error(f"Error deploying Augur: {str(e)}")
    
    with col2:
        st.markdown("### Deployment Status")
        if st.button("Check Status"):
            try:
                core_api = client.CoreV1Api()
                apps_api = client.AppsV1Api()
                
                # Get deployment status
                deployments = apps_api.list_namespaced_deployment(namespace="default")
                pods = core_api.list_namespaced_pod(namespace="default")
                services = core_api.list_namespaced_service(namespace="default")
                
                # Display status
                st.write("Deployments:")
                for dep in deployments.items:
                    st.write(f"- {dep.metadata.name}: {dep.status.ready_replicas}/{dep.status.replicas} ready")
                
                st.write("Pods:")
                for pod in pods.items:
                    st.write(f"- {pod.metadata.name}: {pod.status.phase}")
                
                st.write("Services:")
                for svc in services.items:
                    st.write(f"- {svc.metadata.name}: {svc.spec.type}")
                    
            except Exception as e:
                st.error(f"Error checking status: {str(e)}")
        
        # Add delete button
        if st.button("Delete Deployment"):
            try:
                result = os.system("kubectl delete -f /tmp/augur-manifests.yaml")
                if result == 0:
                    st.success("Augur deployment deleted successfully!")
                else:
                    st.error("Failed to delete Augur deployment.")
            except Exception as e:
                st.error(f"Error deleting deployment: {str(e)}")

def display_infrastructure_details():
    """Display infrastructure details including K8s and container health"""
    st.title("Infrastructure Management")
    
    tab1, tab2 = st.tabs(["Kubernetes Deployment", "Container Health"])
    
    with tab1:
        st.subheader("Kubernetes Deployment Status")
        visualize_kubernetes_deployment()
    
    with tab2:
        st.subheader("Container Health Monitor")
        monitor_container_health()

def is_running_on_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud"""
    try:
        # Streamlit Cloud sets this environment variable
        return os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'
    except:
        return False

def monitor_container_health():
    """Monitor Docker container health and logs"""
    st.subheader("Container Health Monitor")
    
    if is_running_on_streamlit_cloud():
        st.info("📢 Running on Streamlit Cloud - showing demo data")
        show_demo_container_data()
        return
        
    # Local development with Docker
    try:
        client = docker.from_env()
        client.ping()
        containers = client.containers.list(all=True)
        
        if not containers:
            st.warning("No containers found")
            if st.button("Show Demo Data"):
                show_demo_container_data()
            return
            
        # Show real container data
        for container in containers:
            st.markdown(f"### 📦 {container.name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", container.status)
            with col2:
                st.metric("Health", container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A'))
            with col3:
                if st.button(f"Restart {container.name}", key=container.name):
                    try:
                        container.restart()
                        st.success(f"Container {container.name} restarted successfully")
                    except Exception as e:
                        st.error(f"Failed to restart container: {str(e)}")
            
            # Logs
            st.markdown("**Logs:**")
            try:
                logs = container.logs(tail=100, timestamps=True).decode('utf-8')
                st.code(logs)
            except Exception as e:
                st.error(f"Error fetching logs: {str(e)}")
            
            st.markdown("---")
            
    except Exception as e:
        st.error("Docker is not accessible")
        if st.button("Show Demo Data"):
            show_demo_container_data()

def show_demo_container_data():
    """Show demo container data"""
    demo_containers = [
        {
            'name': 'augur-frontend',
            'status': 'running',
            'health': 'healthy',
            'logs': 'Frontend service running on port 5000\nConnected to backend service\n',
            'memory': '256MB',
            'cpu': '0.5%',
            'uptime': '2 days',
            'version': 'latest'
        },
        {
            'name': 'augur-backend',
            'status': 'running',
            'health': 'healthy',
            'logs': 'Backend API started\nDatabase connection established\n',
            'memory': '512MB',
            'cpu': '1.2%',
            'uptime': '2 days',
            'version': 'latest'
        },
        {
            'name': 'augur-db',
            'status': 'running',
            'health': 'healthy',
            'logs': 'PostgreSQL database running\nAccepting connections\n',
            'memory': '1GB',
            'cpu': '0.8%',
            'uptime': '2 days',
            'version': '14.5'
        }
    ]
    
    # Overall system metrics
    st.success("✅ Showing simulated container status")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Containers", len(demo_containers))
    with col2:
        st.metric("Healthy Containers", len([c for c in demo_containers if c['health'] == 'healthy']))
    with col3:
        st.metric("Total Memory", "1.75GB")
    with col4:
        st.metric("Total CPU", "2.5%")

    # Container details
    st.subheader("Container Details")
    for container in demo_containers:
        st.markdown(f"### 📦 {container['name']} ({container['status']})")
        
        # Container metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Memory Usage", container['memory'])
        with col2:
            st.metric("CPU Usage", container['cpu'])
        with col3:
            st.metric("Health", container['health'])
            if container['health'] == 'healthy':
                st.success("✅ Healthy")
            else:
                st.error("❌ Unhealthy")
        
        # Container info
        st.markdown(f"**Version:** {container['version']} | **Uptime:** {container['uptime']}")
        
        # Logs
        st.markdown("**Logs:**")
        st.code(container['logs'])
        
        # Actions
        if st.button(f"Restart {container['name']}", key=container['name']):
            st.info(f"Demo: Simulated restart of {container['name']}")
        
        st.markdown("---")  # Separator between containers

    # Add a note about demo mode
    st.markdown("""
    ℹ️ **Note:** This is a demo visualization showing simulated container data.
    In a real deployment, you would see:
    - Real-time container metrics
    - Actual container logs
    - Live container status
    - Working container controls
    """)


def display_infrastructure_details():
    """Display infrastructure details including K8s and container health"""
    st.title("Infrastructure Management")
    
    tab1, tab2 = st.tabs(["Kubernetes Deployment", "Container Health"])
    
    with tab1:
        st.subheader("Kubernetes Deployment Status")
        fig = visualize_kubernetes_deployment()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Container Health Monitor")
        monitor_container_health()


def create_mini_8knot():
    st.set_page_config(page_title="Mini 8Knot", layout="wide")
    
    # Sidebar for repository selection and filters
    st.sidebar.title("8Knot Controls")
    repo_name = st.sidebar.text_input("Repository", value="chaoss/augur")
    date_range = st.sidebar.slider(
        "Date Range",
        min_value=datetime.now() - timedelta(days=365*2),  # Allow up to 2 years of data
        max_value=datetime.now(),
        value=(datetime.now() - timedelta(days=365), datetime.now())
    )
    
    #token = ''  # Replace with your actual token
    commit_df, contributors, contributions, code_changes = fetch_github_data(repo_name, date_range, token)
    prs_data, issues_data, num_merged_prs, avg_time_to_merge, num_open_issues, num_closed_issues, stars, forks, contributors_data = fetch_additional_insights(repo_name, token)
    
    # Calculate daily commits
    commit_df['date'] = pd.to_datetime(commit_df['date'])
    daily_commits = commit_df.groupby(commit_df['date'].dt.date).size()


    # Main layout
    st.title("Mini 8Knot - Open Source Analytics")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Contributors", "Commit Activity & Networks", "Metrics & PRs",
        "Transformer-Based Chatbot For PRs, Issues & Contributors", 
        "CI/CD Integration", "Issue Prediction (Machine Learning model)", 
        "Container Infrastructure", "Infrastructure Management"
        ])
    

    with tab1:
        st.subheader("Contributor Analysis")
        
        # Create contributor plot
        if contributors:
            fig_contributors = px.bar(
                x=contributors,
                y=contributions,
                title="Top Contributors",
                labels={'x': 'Contributor', 'y': 'Contributions'}
            )
            st.plotly_chart(fig_contributors, use_container_width=True)
        else:
            st.warning("No contributors data available.")
        
        # Contributor metrics
        st.metric("Total Contributors", len(contributors))
         
    with tab2:
        st.subheader("Commit Activity Heatmap & Networks")
        
        # Heatmap for commit activity
        if not daily_commits.empty:
            heatmap_data = pd.DataFrame({
                'Date': daily_commits.index,
                'Commits': daily_commits.values
            })
            fig_heatmap = px.density_heatmap(heatmap_data, x='Date', y='Commits', title="Commit Activity Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No data available for heatmap.")
        
        # Real interaction network
        if contributors:
            fig_real_network = create_real_interaction_network(contributors, commit_df)
            st.plotly_chart(fig_real_network, use_container_width=True)
        else:
            st.warning("No data available for real interaction network.")
        
        # Example network with random connections
        if contributors:
            fig_example_network = create_example_network(contributors, contributions)
            st.plotly_chart(fig_example_network, use_container_width=True)
        else:
            st.warning("No data available for example network.")
    
    with tab3:
        st.subheader("Metrics & PRs to Review")
        
        # Display additional insights
        st.metric("Merged PRs", num_merged_prs)
        st.metric("Average Time to Merge (days)", avg_time_to_merge)
        st.metric("Open Issues", num_open_issues)
        st.metric("Closed Issues", num_closed_issues)
        st.metric("Stars", stars)
        st.metric("Forks", forks)
        
        # Display CHAOSS metrics
        display_chaoss_metrics(commit_df, num_open_issues, num_closed_issues)
        
        # Provide prescriptive recommendations
        recommendations, unreviewed_prs, stale_issues = provide_recommendations(prs_data, issues_data)
        for rec in recommendations:
            st.write(rec)
        
        # Dropdown for unreviewed PRs
        if unreviewed_prs:
            pr_titles = [pr['title'] for pr in unreviewed_prs]
            selected_pr = st.selectbox("PRs to review", pr_titles)
            st.write(f"Selected PR: {selected_pr}")
        
        # Dropdown for stale issues
        if stale_issues:
            issue_titles = [issue['title'] for issue in stale_issues]
            selected_issue = st.selectbox("Stales issues to address", issue_titles)
            st.write(f"Selected Issue: {selected_issue}")

    with tab4:
        st.subheader("Transformer-Based Chatbot For PRs, Issues & Contributors")
        stars, forks, open_issues = fetch_github_repo_stats(repo_name, token)
        st.metric("Stars", stars)
        st.metric("Forks", forks)
        st.metric("Open Issues", open_issues)

        # NLP Query
        query = st.text_input("Ask a question about PRs, issues, or contributors:")
        if query:
            prs_data, issues_data, num_merged_prs, avg_time_to_merge, num_open_issues, num_closed_issues, stars, forks, contributors_data = fetch_additional_insights(repo_name, token)
            results = process_query(query, prs_data, issues_data, contributors_data)
            st.write("Results:", results)

    with tab5:
        st.subheader("CI/CD Integration")
        run_details = fetch_github_actions_status(repo_name, token)
        if run_details:
            df_runs = pd.DataFrame(run_details)
            st.write("Latest 20 Workflow Runs", df_runs)
        else:
            st.warning("No workflow runs available.")
    
    with tab6:
        st.subheader("Issue Analysis & Prediction")
        
        # Prepare and train model
        issues_df = prepare_issue_data(issues_data)
        
        # Display issue statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", len(issues_df))
        with col2:
            st.metric("Stale Issues", sum(issues_df['is_stale']))
        with col3:
            stale_rate = (sum(issues_df['is_stale']) / len(issues_df)) * 100
            st.metric("Stale Rate", f"{stale_rate:.1f}%")
        
        # Train model and save
        if not issues_df.empty:
            model, vectorizer = train_issue_model(issues_df)
            
            # Issue prediction form
            st.subheader("Predict Issue Status")
            new_title = st.text_input("Issue Title")
            new_body = st.text_area("Issue Description")
            new_comments = st.number_input("Number of Comments", min_value=0)
            new_has_labels = st.checkbox("Has Labels")
            
            if st.button("Predict"):
                if new_title:
                    probability = predict_issue_status(
                        model, vectorizer, new_title, new_body, 
                        new_comments, new_has_labels
                    )
                    st.write(f"Probability of becoming stale: {probability:.1%}")
                    
                    if probability < 0.3:
                        st.success("Low risk of becoming stale")
                    elif probability < 0.7:
                        st.warning("Medium risk of becoming stale")
                    else:
                        st.error("High risk of becoming stale")
                else:
                    st.error("Please enter at least a title")
            
            # Display current stale issues
            st.subheader("Current Stale Issues")
            stale_issues_df = issues_df[issues_df['is_stale']]
            if not stale_issues_df.empty:
                st.dataframe(stale_issues_df[['title', 'created_at', 'comments']])
            else:
                st.info("No stale issues found")
        else:
            st.warning("No issue data available for analysis")

    with tab7:
        display_cicd_details()    

    with tab8:
        display_infrastructure_details()   
    
    
    # Footer with information about the project
    st.markdown("""
    ---
    ### About Mini 8Knot
    This is a simplified version of the [8Knot project](https://github.com/oss-aspen/8Knot/tree/dev) by Red Hat's Open Source Program Office (OSPO).
    The original project provides comprehensive analytics for Open Source communities using data from Augur.
    """)

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

def fetch_github_actions_status(repo_name, token):
    headers = {'Authorization': f'token {token}'}
    actions_url = f'https://api.github.com/repos/{repo_name}/actions/runs'
    
    # Fetch latest workflow runs
    actions_response = requests.get(actions_url, headers=headers)
    actions_data = actions_response.json()
    
    runs = actions_data.get('workflow_runs', [])
    latest_runs = runs[:10]  # Get the latest 10 runs
    
    run_details = []
    for run in latest_runs:
        run_info = {
            'Name': run.get('name', 'N/A'),
            'Status': run.get('conclusion', 'unknown'),
            'Duration (min)': run.get('run_duration_ms', 0) / 60000,  # Convert ms to minutes
            'Created At': run.get('created_at', 'N/A')
        }
        run_details.append(run_info)
    
    return run_details



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
    
    # Main layout
    st.title("Mini 8Knot - Open Source Analytics")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Contributors", "Custom Analysis", "Commit Activity & Networks", "Metrics & PRs", "Live Dashboard", "CI/CD Integration"])
    

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
        st.subheader("Custom Analysis with Interactive Plotly")
        comprehensive_df = create_comprehensive_dataset(commit_df, contributors, prs_data, issues_data)
        
        # Display the DataFrame
        st.write("Contribution DataFrame & PRs/Issues", comprehensive_df)
        
        # Create an interactive bar chart
        if not comprehensive_df.empty:
            fig = px.bar(comprehensive_df, x='Type', y='Days Open', color='state', barmode='group',
                         title="Days Open by Type and State")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for custom analysis.")
    
    
    with tab3:
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
    
    with tab4:
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

    with tab5:
        st.subheader("Live Dashboard")
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

    with tab6:
        st.subheader("CI/CD Integration")
        run_details = fetch_github_actions_status(repo_name, token)
        if run_details:
            df_runs = pd.DataFrame(run_details)
            st.write("Latest 10 Workflow Runs", df_runs)
        else:
            st.warning("No workflow runs available.")
    
    
    
    # Footer with information about the project
    st.markdown("""
    ---
    ### About Mini 8Knot
    This is a simplified version of the [8Knot project](https://github.com/oss-aspen/8Knot/tree/dev) by Red Hat's Open Source Program Office (OSPO).
    The original project provides comprehensive analytics for Open Source communities using data from Augur.
    """)

if __name__ == "__main__":
    create_mini_8knot()

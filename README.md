# Mini 8Knot - Open Source Analytics
Created using: 8Knot as reference, and developed using Cursor AI IDE with Claude.Sonnet3.5 and GPT4o

Mini 8Knot is a Streamlit application designed to provide analytics for open source projects hosted on GitHub. It offers insights into contribution activity, contributor interactions, code changes, and more, using data fetched from the GitHub API.

![image](https://github.com/user-attachments/assets/67fb95bf-78a2-4663-b27a-21500b48f3c4)

![image](https://github.com/user-attachments/assets/565b6a77-2478-46c4-9ef5-b8a3b775bc77)

![image](https://github.com/user-attachments/assets/bb0aaef7-0971-4bf2-8068-49991520cd03)

![image](https://github.com/user-attachments/assets/b28a82bb-cad9-4e78-8e6e-5c2f5d056abc)

## Features

- **Contributors Analysis**: Visualize top contributors and their contributions.
- **Custom Analysis**: Interactive analysis of pull requests and issues.
- **Commit Activity & Networks**: Heatmap of commit activity and interaction networks.
- **Metrics & PRs**: Insights into pull requests, issues, and repository metrics.
- **Live Dashboard**: Real-time repository statistics and NLP-based query handling.
- **CI/CD Integration**: Display latest GitHub Actions workflow runs.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mini-8knot.git
   cd mini-8knot
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your GitHub Token**:
   Replace `'your_github_token_here'` in the code with your actual GitHub token.

## Usage

Run the application using Streamlit:

bash

streamlit run mini_8Knot.py


## Configuration

- **Repository Selection**: Use the sidebar to input the GitHub repository name (e.g., `chaoss/augur`).
- **Date Range**: Adjust the date range for data analysis.

## Code Overview

### Key Functions

- **fetch_github_data**: Retrieves commit data, contributors, and code changes.
- **fetch_github_repo_stats**: Fetches repository statistics like stars and forks.
- **fetch_github_actions_status**: Retrieves the latest GitHub Actions workflow runs.
- **fetch_additional_insights**: Gathers data on pull requests, issues, and contributors.
- **process_query**: Uses NLP to interpret user queries about PRs, issues, or contributors.
- **create_comprehensive_dataset**: Combines data into a comprehensive DataFrame for analysis.
- **provide_recommendations**: Offers recommendations based on PR and issue data.
- **create_real_interaction_network**: Visualizes contributor interactions as a network graph.

### Tabs

- **Contributors**: Displays a bar chart of top contributors.
- **Custom Analysis**: Provides an interactive bar chart of PRs and issues.
- **Commit Activity & Networks**: Shows a heatmap of commit activity and interaction networks.
- **Metrics & PRs**: Displays repository metrics and recommendations for PRs and issues.
- **Live Dashboard**: Offers real-time stats and NLP query handling.
- **CI/CD Integration**: Lists the latest 10 GitHub Actions workflow runs.

## Requirements

- Python 3.7+
- Streamlit
- Plotly
- Pandas
- NumPy
- NetworkX
- Requests
- Transformers

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

This project is inspired by the [8Knot project](https://github.com/oss-aspen/8Knot/tree/dev) by Red Hat's OSPO.

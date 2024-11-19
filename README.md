# Mini 8Knot - Open Source Analytics
Created using: 8Knot as reference, and developed using Cursor AI IDE with Claude.Sonnet3.5 and GPT4o

Mini 8Knot is a Streamlit application designed to provide analytics for open source projects hosted on GitHub. It offers insights into contribution activity, contributor interactions, code changes, and more, using data fetched from the GitHub API.

## Features

- **Contribution Activity**: Visualize daily commit activity over a specified date range.
- **Contributor Analysis**: Analyze top contributors and their contributions.
- **Code Changes**: Overview of code changes including additions, deletions, and files changed.
- **Custom Analysis**: Use Pygwalker for custom data exploration.
- **Insights**: 
  - Real and example interaction networks of contributors.
  - CHAOSS metrics such as issue closure rate.
  - Prescriptive recommendations for repository maintainers.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mini-8knot.git
   cd mini-8knot
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or later installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up GitHub Token**:
   Replace `'your_github_personal_access_token'` in the script with your actual GitHub personal access token to authenticate API requests.

## Usage

1. **Run the Application**:
   ```bash
   streamlit run 8knot_tableau.py
   ```

2. **Interact with the Dashboard**:
   - Use the sidebar to select a GitHub repository and specify a date range.
   - Explore different tabs for various analytics and insights.

## Requirements

- Python 3.7+
- Streamlit
- Plotly
- Pandas
- NumPy
- NetworkX
- Pygwalker
- Requests
- python-louvain

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This application is inspired by the [8Knot project](https://github.com/oss-aspen/8Knot/tree/dev) by Red Hat's Open Source Program Office (OSPO), which provides comprehensive analytics for open source communities using data from Augur.

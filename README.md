# Azure Usage Analysis

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
<!-- Add build status and Docker badges if available -->

A Python web application that analyzes Azure usage data from a CSV file and displays interactive visualizations through a Flask-powered web interface. Features optional AI-powered insights using Claude AI.

---

## Quick Start

```bash
git clone https://github.com/your-org/azure-usage-analysis.git
cd azure-usage-analysis
cp .env.example .env  # Edit .env with your values
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate or venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py  # or python app.py
```

Open your browser at [http://localhost:9099](http://localhost:9099)

---

## Features

- **Data Analysis**: Processes Azure usage CSV data and identifies the top 5 most expensive services per week
- **Interactive Dashboard**: Visualizes the data using both static (Matplotlib) and interactive (Plotly) charts
- **API Endpoint**: Provides the analyzed data in JSON format through a RESTful API endpoint
- **Responsive Design**: Mobile-friendly UI built with Bootstrap
- **Optional AI Insights**: Claude AI integration for cost optimization suggestions

---

## Screenshots

<!-- Add screenshots or GIFs here -->

---

## Project Structure

```
azure-usage-analysis/
├── app.py                 # Main Flask application
├── run.py                 # Alternate entry point
├── test_app.py            # Unit tests for the application
├── requirements.txt       # Project dependencies
├── data/
│   └── AzureUsage.csv     # Azure usage data (CSV format)
├── templates/
│   └── index.html         # HTML template for the web interface
└── static/                # Static files (CSS, JS, images)
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Never commit your real `.env` file or secrets to the repository.

| Variable                        | Required | Description                                 | Example Value                |
|----------------------------------|----------|---------------------------------------------|------------------------------|
| PORT                             | No       | Port to run the app on                      | 9099                         |
| FLASK_ENV                        | No       | Flask environment (development/production)   | development                  |
| SECRET_KEY                       | Yes      | Flask secret key                            | changeme                     |
| AZURE_STORAGE_CONNECTION_STRING  | No       | Azure Blob Storage connection string         | DefaultEndpointsProtocol=... |
| AZURE_STORAGE_CONTAINER_NAME     | No       | Azure Blob Storage container name            | uploads                      |
| APPINSIGHTS_INSTRUMENTATIONKEY   | No       | Azure Application Insights key               | 00000000-0000-0000-0000-...  |
| CLAUDE_API_KEY                   | No       | Claude AI API key for insights               | sk-...                       |
| UPLOAD_FOLDER                    | No       | Local upload directory                       | uploads                      |
| DATA_FOLDER                      | No       | Local data directory                         | data                         |

---

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate or venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and edit as needed.
5. Place your Azure usage CSV file in the `data/` directory and name it `AzureUsage.csv` (or set the path in `.env`).
6. Run the application:
   ```bash
   python run.py
   # or
   python app.py
   ```
7. Open [http://localhost:9099](http://localhost:9099) in your browser.

---

## CSV Format Requirements

The application expects a CSV file with at least the following columns:
- `Date`: The date of the usage (in a format that can be parsed by pandas)
- `ServiceName`: The name of the Azure service
- `Cost`: The cost value (numerical)

---

## Testing

To run the tests:
```bash
python test_app.py
```

Tests cover:
- Data processing pipeline
- Malformed CSV handling
- Corrupt date handling
- File not found handling

---

## API Usage

The application provides a REST API endpoint to get the analysis data in JSON format:
```
GET /api/data
```

---

## Deployment

### Azure App Service (Recommended)

1. Create an Azure App Service with Python 3.9+ runtime.
2. Configure deployment credentials and environment variables (see table above).
3. Deploy your code (via Git, GitHub Actions, or FTP).
4. Set up Azure Blob Storage and Application Insights as needed.

### Docker

1. Build the Docker image:
   ```bash
   docker build -t azure-usage-analysis:latest .
   ```
2. Run the container:
   ```bash
   docker run --env-file .env -p 9099:9099 azure-usage-analysis:latest
   ```

### GitHub Actions (CI/CD)

- Fork the repo and set up your own secrets for deployment.
- Store your Azure App Service publish profile as a GitHub secret named `AZURE_WEBAPP_PUBLISH_PROFILE`.

---

## Security

- **Never commit your real `.env` file or secrets to the repository.**
- Use environment variables for all sensitive data.
- Claude AI integration is optional; the app works without it.

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to your fork and open a pull request

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and write clear commit messages.

---

## License

MIT

---

## FAQ

**Q: Why is my chart blank?**
A: Check that your CSV file has data and is in the correct format.

**Q: How do I reset the app?**
A: Stop the server, clear the `uploads/` directory, and restart.

**Q: Can I use this without Azure Blob Storage or Claude AI?**
A: Yes! Both are optional.

---

## Acknowledgements

This project was built as a demonstration of data analysis and visualization with Python. Special thanks to the open source community and all contributors! 
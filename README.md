---
title: DocumentQA
emoji: 🏃
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.14.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# DocumentQA

This project provides a custom chatbot for interacting with PDF files. It allows you to do question and answer with your documents while providing useful summaries.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

- [Python](https://www.python.org/downloads/) (version 3.10.4)
- [Virtualenv](https://virtualenv.pypa.io/en/stable/) (optional but recommended)

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/LVKinyanjui/DocumentQA
    ```

2. Navigate to the project directory:

    ```bash
    cd DocumentQA
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```


## Obtaining API Keys
### Prerequisites

Before you begin, ensure that you have the following prerequisites:

- [Pinecone Account](https://pinecone.io/signup): Sign up for a Pinecone account to access the Pinecone Vector Database.
- [Google Cloud Account](https://console.cloud.google.com/): Create a Google Cloud account and set up a project to use the Google Vertex AI API.


### Pinecone Vector Database

Follow these steps to obtain your Pinecone API keys:

1. **Sign in:** Log in to your Pinecone account at [Pinecone Console](https://pinecone.io/console).

2. **Create a Project:** If you don't have a project, create one from the Pinecone console.

3. **Generate API Keys:** In your project settings, navigate to the "API Keys" section and generate a new API key.

4. **Copy Keys:** Once generated, copy the Pinecone API key for use in your project. Keep it secure.

### Google Vertex API

Follow these steps to obtain your Google Vertex API keys:

1. **Enable Vertex AI API:** In the [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Either **Create API Key in New Project** or **Create API Key in Ezisting Project** if you had already created a project in Google Cloud.

3. Copy the generated API key.

## Security Considerations

- **Keep Keys Secure:** Treat both Pinecone and Google Vertex API keys like passwords. Keep them confidential and avoid exposing them in public repositories.

- **Use Environment Variables:** Consider using environment variables to store and access both API keys securely.

- **Different Environments:** Be mindful of using separate keys for development, testing, and production environments.


### Running the Application

Make sure you are in the project directory and your virtual environment is activated.
You will also ensure your API keys are set in the credentials file.


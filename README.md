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

### Running the Application

Make sure you are in the project directory and your virtual environment is activated.

Run the following command to start the application:

```bash
python app.py
```


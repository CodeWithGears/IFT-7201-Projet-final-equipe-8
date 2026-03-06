# FrozenLake RL Project

A simple reinforcement learning project based on the Frozen Lake environment.

## Setup

1. **Clone the repository:**
   ```
   git clone <your-repo-url>
   cd frozenlake-rl
   ```
2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
frozenlake-rl/
│
├── README.md
├── .gitignore
├── requirements.txt
├── venv/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── env.py
│   └── agent.py
│
├── tests/
│   └── test_agent.py
```

## Usage

- Add your RL code in `src/`.
- Run experiments from `src/main.py`.
- Add tests in `tests/`.

## License

MIT License

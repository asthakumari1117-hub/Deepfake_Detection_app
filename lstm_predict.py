import subprocess

def run_prediction():

    result = subprocess.check_output(
        ["python", "lstm_inference.py"],
        text=True
    )

    return result
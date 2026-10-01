from openai import OpenAI
from dotenv import load_dotenv

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()

def main():
    client = OpenAI(
        base_url=endpoint
    )

    audio_file = open("sample.mp3", "rb")

    transcription = client.audio.transcriptions.create(
        model=deployment_name,
        file=audio_file
    )

    print("Transcription:")
    print(transcription.text)


if __name__ == "__main__":
    main()
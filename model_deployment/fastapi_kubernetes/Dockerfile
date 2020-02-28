FROM python:3.6

EXPOSE 80

COPY ./app /app

RUN pip install -r /app/requirements.txt

# under the app folder, we have the main.py file, which contains the app object
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

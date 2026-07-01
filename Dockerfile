FROM python:3.9.0

WORKDIR /home/app

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

# CMD python app.py
CMD fastapi run app.py
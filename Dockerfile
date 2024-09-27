FROM python:3.12

# Dependencies Intall and Update
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow[and-cuda] # Instalar tensorflow con soporte para GPU

RUN pip install --upgrade pip

# Instalar dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Workdir
WORKDIR /lab

# Copiar el notebook al contenedor
COPY ./src .

# Exponer el puerto para Jupyter Notebook
EXPOSE 8888

# Comando para ejecutar Jupyter Notebook y mantener el contenedor corriendo
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root && tail -f /dev/null"]

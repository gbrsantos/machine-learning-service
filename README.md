# Machine Learning

Este pequeno projeto faz parte do MVP da Disciplina Qualidade de Software, segurança e sistemas inteligente.

# Aplicação de machine learning em python
Este projeto é uma aplicação de aprendizado de máquina supervisionado de classificação. É possível obter uma previsão das saídas de acordo com o dataset fornecido.

# Bibliotecas
- scikit-learn
- numpy
- flask
- sqlAlchemy
- matplotlib

# Testes

O projeto inclui testes das saídas para o modelo que foi desenvolvido.
````
pytest -v test_modelos.py
````

---
## Como executar 

> Antes, garanta ter instalado a versão mais recentes do [docker](https://docs.docker.com/desktop/install/windows-install/).

```
docker-compose build
```
Este comando instala a aplicação.

```
docker-compose up
```

Este comando roda a aplicação.

Abra o [http://localhost:5000/#/](http://localhost:5000/#/) no navegador para verificar o status da API em execução.

-----

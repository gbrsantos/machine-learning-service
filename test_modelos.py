import json
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from app import processar
#pytest -v test_modelos.py

# Método para testar a função de avaliação do modelo de teste
def test_processar():
   
    url = "https://raw.githubusercontent.com/gbrsantos/datascience/main/bankloan.csv"
    dataset = pd.read_csv(url, delimiter=',')

    inputs = [{"name":"ID","values":"1,2,3"},
            {"name":"Age","values":"34, 42, 30"},
            {"name":"Experience","values":"9, 12, 12"},
            {"name":"Income","values":"110, 30, 75"},
            {"name":"ZIP.Code","values":"1234, 12345, 123456"},
            {"name":"Family","values":"1, 2, 2"},
            {"name":"CCAvg","values":"8.9, 1, 0.3"},
            {"name":"Education","values":"0, 2, 3"},
            {"name":"Mortgage","values":"0, 150, 0"},
            {"name":"Securities.Account","values":"0, 1, 0"},
            {"name":"CD.Account","values":"3, 0, 0"},
            {"name":"Online","values":"0, 0, 1"},
            {"name":"CreditCard","values":"0, 0, 1"}]
    json_string = json.dumps(inputs, indent=2)


    saidas = processar(dataset=dataset, campo_saida= 'Personal.Loan', inputs = json_string)
   
    assert isinstance(saidas, list)
    assert saidas == [1,0,0]
   
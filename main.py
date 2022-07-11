import sys
import pandas as pd
import numpy as np
import datetime
import random
import csv
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from enum import Enum
import logging
import logging.config
import sklearn.metrics as skmetrics
import scipy
from sklearn import svm
from sklearn.model_selection import StratifiedKFold


logging.config.fileConfig("loggerconfig.ini")
logger = logging.getLogger(name="ProcessaSVM")

class ProcessContent(Enum):
    """
    Todas as colunas
    """
    ALL = 1
    PCA = 2
    SELECT01 = 3
    SELECT02 = 4

class ProcessaSVM_old_version:
    path_root: str
    processType: ProcessContent
    processCode: str
    plano: list = []
    path_controle: str
    controle_json: dict

    def __init__(self, mytype: ProcessContent, path_root: str = None):
        self.processType = mytype
        self.processCode = f"{mytype.value:02d}"
        self.path_root = path_root or "/Users/elthon.freitas/Google Drive/Shared drives/SIN5007-2022"
        self.path_controle = f"{self.path_root}/Experimentos/svm_dataset{self.processCode}.json"

    def generate_plano(self) -> list:
        alphas = [0.1, 0.5, 1.0]
        C = [0.1, 0.5, 1.0]
        kernel = ['linear', 'rbf', 'sigmoid']  # 'linear', 'rbf', 'poly', 'sigmoid'
        poly_degree = [2, 3, 4]
        gamma_poly_rbf = ['scale', 'auto', 0.1, 1.0, 10]

        self.plano = []
        for c in C:
            for k in kernel:
                if k in ['poly', 'rbf', 'sigmoid']:
                    for g in gamma_poly_rbf:
                        if k in ['poly']:
                            for d in poly_degree:
                                self.plano.append({"kernel": k, "C": c, "degree": d, "gamma": g, 'max_iter': 5e5})
                        else:
                            self.plano.append({"kernel": k, "C": c, "degree": 3, "gamma": g, 'max_iter': 5e5})
                else:
                    self.plano.append({"kernel": k, "C": c, "degree": 3, "gamma": 'scale', 'max_iter': 5e5})
        return self.plano

    def controle_read(self):
        path_controle = self.path_controle

        if not Path(path_controle).is_file():
            logger.info("Arquivo controle não existe. O mesmo será criado.")
            controle_json = {"experimento": {}}
            _ = self.generate_plano()
            for i, p in enumerate(self.plano):
                controle_json["experimento"][str(i)] = {"parametros": p,
                                                        "resultados": None}
            Path(path_controle).parent.mkdir(parents=True, exist_ok=True)
            with open(path_controle, 'w') as outfile:
                json.dump(controle_json, outfile, indent=4)
            self.controle_json = controle_json
            return self.controle_json
        else:
            logger.info("Arquivo controle existente. O mesmo será carregado.")
            with open(path_controle, 'r') as impfile:
                controle_json = json.load(impfile)
            self.controle_json = controle_json
            return controle_json

    def controle_write(self, content: dict):
        path_controle = self.path_controle
        with open(path_controle, 'w') as outfile:
            json.dump(content, outfile, indent=4)
        pass

    def get_dataset(self, percent: int = 1, normaliza: bool = False) -> tuple:
        X = None
        y = None
        if percent < 1 or percent > 100:
            raise NotImplementedError()
        percent_filter = f"{percent-1:02d}"

        pathPreprocpq = f"{self.path_root}/Datasets/PREPROC/PreProcessamento01.parquet"
        pathPCApq = f"{self.path_root}/Datasets/PREPROC/PCA01.parquet"

        #Todas as colunas após Pré-processamento
        colunasALL = ['IN_SURDEZ', 'IN_MARCA_PASSO', 'Q015', 'Q005', 'IN_MAQUINA_BRAILE', 'Q014', 'IN_MOBILIARIO_ESPECIFICO', 'IN_GESTANTE', 'Q021', 'IN_AMPLIADA_18', 'IN_AMPLIADA_24', 'IN_DEFICIENCIA_AUDITIVA', 'Q012', 'IN_SEM_RECURSO', 'IN_MESA_CADEIRA_SEPARADA', 'IN_SALA_INDIVIDUAL', 'IN_SONDA', 'IN_ESTUDA_CLASSE_HOSPITALAR', 'IN_TREINEIRO', 'Q011', 'IN_PROVA_DEITADO', 'IN_SALA_ESPECIAL', 'Q024', 'TP_NACIONALIDADE', 'IN_MOBILIARIO_OBESO', 'Q022', 'IN_APOIO_PERNA', 'IN_LACTANTE', 'IN_DEFICIT_ATENCAO', 'IN_COMPUTADOR', 'Q010', 'IN_CADEIRA_ACOLCHOADA', 'Q020', 'IN_DISCALCULIA', 'IN_LIBRAS', 'IN_SALA_ACOMPANHANTE', 'Q007', 'IN_TEMPO_ADICIONAL', 'IN_LEDOR', 'IN_NOME_SOCIAL', 'IN_GUIA_INTERPRETE', 'Q006', 'IN_DEFICIENCIA_FISICA', 'IN_MATERIAL_ESPECIFICO', 'IN_TRANSCRICAO', 'Q016', 'IN_DEFICIENCIA_MENTAL', 'IN_CADEIRA_ESPECIAL', 'IN_BRAILLE', 'IN_SURDO_CEGUEIRA', 'IN_LEITURA_LABIAL', 'Q019', 'Q017', 'IN_OUTRA_DEF', 'IN_IDOSO', 'Q003', 'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'Q013', 'Q025', 'Q023', 'TP_SEXO', 'IN_SOROBAN', 'IN_MESA_CADEIRA_RODAS', 'IN_MEDIDOR_GLICOSE', 'Q004', 'IN_DISLEXIA', 'Q001', 'IN_LAMINA_OVERLAY', 'Q002', 'IN_ACESSO', 'IN_BAIXA_VISAO', 'IN_MEDICAMENTOS', 'Q009', 'IN_PROTETOR_AURICULAR', 'Q018', 'IN_CADEIRA_CANHOTO', 'Q008', 'IN_CEGUEIRA', 'NU_INSCRICAO', 'TP_FAIXA_ETARIA', 'TOPMATH', 'TP_DEPENDENCIA_ADM_ESC_0', 'TP_DEPENDENCIA_ADM_ESC_1', 'TP_DEPENDENCIA_ADM_ESC_2', 'TP_DEPENDENCIA_ADM_ESC_3', 'TP_DEPENDENCIA_ADM_ESC_4', 'TP_ESTADO_CIVIL_1', 'TP_ESTADO_CIVIL_2', 'TP_ESTADO_CIVIL_3', 'TP_ESTADO_CIVIL_4', 'TP_LOCALIZACAO_ESC_0', 'TP_LOCALIZACAO_ESC_1', 'TP_LOCALIZACAO_ESC_2', 'TP_ANO_CONCLUIU_0', 'TP_ANO_CONCLUIU_1', 'TP_ANO_CONCLUIU_2', 'TP_ANO_CONCLUIU_3', 'TP_ANO_CONCLUIU_4', 'TP_ANO_CONCLUIU_5', 'TP_ANO_CONCLUIU_6', 'TP_ANO_CONCLUIU_7', 'TP_ANO_CONCLUIU_8', 'TP_ANO_CONCLUIU_9', 'TP_ANO_CONCLUIU_10', 'TP_ANO_CONCLUIU_11', 'TP_ANO_CONCLUIU_12', 'TP_ANO_CONCLUIU_13', 'TP_SIT_FUNC_ESC_0', 'TP_SIT_FUNC_ESC_1', 'TP_SIT_FUNC_ESC_2', 'TP_SIT_FUNC_ESC_3', 'TP_SIT_FUNC_ESC_4', 'SG_UF_NASCIMENTO_AC', 'SG_UF_NASCIMENTO_AL', 'SG_UF_NASCIMENTO_AM', 'SG_UF_NASCIMENTO_AP', 'SG_UF_NASCIMENTO_BA', 'SG_UF_NASCIMENTO_CE', 'SG_UF_NASCIMENTO_DF', 'SG_UF_NASCIMENTO_ES', 'SG_UF_NASCIMENTO_GO', 'SG_UF_NASCIMENTO_MA', 'SG_UF_NASCIMENTO_MG', 'SG_UF_NASCIMENTO_MS', 'SG_UF_NASCIMENTO_MT', 'SG_UF_NASCIMENTO_PA', 'SG_UF_NASCIMENTO_PB', 'SG_UF_NASCIMENTO_PE', 'SG_UF_NASCIMENTO_PI', 'SG_UF_NASCIMENTO_PR', 'SG_UF_NASCIMENTO_RJ', 'SG_UF_NASCIMENTO_RN', 'SG_UF_NASCIMENTO_RO', 'SG_UF_NASCIMENTO_RR', 'SG_UF_NASCIMENTO_RS', 'SG_UF_NASCIMENTO_SC', 'SG_UF_NASCIMENTO_SE', 'SG_UF_NASCIMENTO_SP', 'SG_UF_NASCIMENTO_TO', 'SG_UF_ESC_AC', 'SG_UF_ESC_AL', 'SG_UF_ESC_AM', 'SG_UF_ESC_AP', 'SG_UF_ESC_BA', 'SG_UF_ESC_CE', 'SG_UF_ESC_DF', 'SG_UF_ESC_ES', 'SG_UF_ESC_GO', 'SG_UF_ESC_MA', 'SG_UF_ESC_MG', 'SG_UF_ESC_MS', 'SG_UF_ESC_MT', 'SG_UF_ESC_NA', 'SG_UF_ESC_PA', 'SG_UF_ESC_PB', 'SG_UF_ESC_PE', 'SG_UF_ESC_PI', 'SG_UF_ESC_PR', 'SG_UF_ESC_RJ', 'SG_UF_ESC_RN', 'SG_UF_ESC_RO', 'SG_UF_ESC_RR', 'SG_UF_ESC_RS', 'SG_UF_ESC_SC', 'SG_UF_ESC_SE', 'SG_UF_ESC_SP', 'SG_UF_ESC_TO', 'TP_COR_RACA_1', 'TP_COR_RACA_2', 'TP_COR_RACA_3', 'TP_COR_RACA_4', 'TP_COR_RACA_5', 'SG_UF_RESIDENCIA_AC', 'SG_UF_RESIDENCIA_AL', 'SG_UF_RESIDENCIA_AM', 'SG_UF_RESIDENCIA_AP', 'SG_UF_RESIDENCIA_BA', 'SG_UF_RESIDENCIA_CE', 'SG_UF_RESIDENCIA_DF', 'SG_UF_RESIDENCIA_ES', 'SG_UF_RESIDENCIA_GO', 'SG_UF_RESIDENCIA_MA', 'SG_UF_RESIDENCIA_MG', 'SG_UF_RESIDENCIA_MS', 'SG_UF_RESIDENCIA_MT', 'SG_UF_RESIDENCIA_PA', 'SG_UF_RESIDENCIA_PB', 'SG_UF_RESIDENCIA_PE', 'SG_UF_RESIDENCIA_PI', 'SG_UF_RESIDENCIA_PR', 'SG_UF_RESIDENCIA_RJ', 'SG_UF_RESIDENCIA_RN', 'SG_UF_RESIDENCIA_RO', 'SG_UF_RESIDENCIA_RR', 'SG_UF_RESIDENCIA_RS', 'SG_UF_RESIDENCIA_SC', 'SG_UF_RESIDENCIA_SE', 'SG_UF_RESIDENCIA_SP', 'SG_UF_RESIDENCIA_TO', 'SG_UF_PROVA_AC', 'SG_UF_PROVA_AL', 'SG_UF_PROVA_AM', 'SG_UF_PROVA_AP', 'SG_UF_PROVA_BA', 'SG_UF_PROVA_CE', 'SG_UF_PROVA_DF', 'SG_UF_PROVA_ES', 'SG_UF_PROVA_GO', 'SG_UF_PROVA_MA', 'SG_UF_PROVA_MG', 'SG_UF_PROVA_MS', 'SG_UF_PROVA_MT', 'SG_UF_PROVA_PA', 'SG_UF_PROVA_PB', 'SG_UF_PROVA_PE', 'SG_UF_PROVA_PI', 'SG_UF_PROVA_PR', 'SG_UF_PROVA_RJ', 'SG_UF_PROVA_RN', 'SG_UF_PROVA_RO', 'SG_UF_PROVA_RR', 'SG_UF_PROVA_RS', 'SG_UF_PROVA_SC', 'SG_UF_PROVA_SE', 'SG_UF_PROVA_SP', 'SG_UF_PROVA_TO', 'TP_ESCOLA_1', 'TP_ESCOLA_2', 'TP_ESCOLA_3', 'TP_ENSINO_0', 'TP_ENSINO_1', 'TP_ENSINO_2', 'TP_ST_CONCLUSAO_1', 'TP_ST_CONCLUSAO_2', 'TP_ST_CONCLUSAO_3', 'TP_ST_CONCLUSAO_4', 'DIGITOSISCRICAO']
        #Colunas do Selecionador 1 (Lasso)
        colunasX1 = ['Q002', 'TP_FAIXA_ETARIA', 'Q001', 'Q004', 'Q003', 'Q022', 'Q006', 'Q005', 'Q024', 'TP_SEXO']
        #Colunas do Selecionador 2 (KBest)
        colunasX2 = ['Q003', 'Q004', 'Q002', 'Q024', 'Q001', 'Q006', 'Q019', 'Q008', 'Q018', 'Q010']
        #Colunas do PCA
        colunasXPCATodas = ['NU_INSCRICAO', 'DIGITOSISCRICAO', 'PCA000', 'PCA001', 'PCA002', 'PCA003', 'PCA004', 'PCA005', 'PCA006', 'PCA007', 'PCA008', 'PCA009', 'PCA010', 'PCA011', 'PCA012', 'PCA013', 'PCA014', 'PCA015', 'PCA016', 'PCA017', 'PCA018', 'PCA019', 'PCA020', 'PCA021', 'PCA022', 'PCA023', 'PCA024', 'PCA025', 'PCA026', 'PCA027', 'PCA028', 'PCA029', 'PCA030', 'PCA031', 'PCA032', 'PCA033', 'PCA034', 'PCA035', 'PCA036', 'PCA037', 'PCA038', 'PCA039', 'PCA040', 'PCA041', 'PCA042', 'PCA043', 'PCA044', 'PCA045', 'PCA046', 'PCA047', 'PCA048', 'PCA049']
        colunasXPCA = ['NU_INSCRICAO', 'DIGITOSISCRICAO', 'PCA000', 'PCA001', 'PCA002', 'PCA003']

        colunasID = ["DIGITOSISCRICAO", 'NU_INSCRICAO']
        colunaY = 'TOPMATH'

        if self.processType == ProcessContent.ALL:
            colunas = set(colunasALL).union(set([colunaY])).difference(set(colunasID))
            dfDados = pd.read_parquet(pathPreprocpq,
                                      columns=colunas,
                                      filters=[
                                          ("DIGITOSISCRICAO", "<=", percent_filter),
                                      ]
                                      )
            X = np.array(dfDados[set(colunasALL).difference(set(colunasID)).difference(set([colunaY]))])
            y = np.array(dfDados[colunaY])
        elif self.processType == ProcessContent.PCA:
            colunas = set(colunasID).union(set([colunaY]))
            dfDados = pd.read_parquet(pathPreprocpq,
                                      columns=colunas,
                                      filters=[
                                          ("DIGITOSISCRICAO", "<=", percent_filter),
                                      ]
                                      )
            dfDadosPCA = pd.read_parquet(pathPCApq,
                                         columns=set(colunasID).union(set(colunasXPCA)),
                                         filters=[
                                             ("DIGITOSISCRICAO", "<=", percent_filter),
                                         ]
                                         )
            dfDados = pd.merge(dfDados, dfDadosPCA)
            X = np.array(dfDados[set(colunasXPCA).difference(set(colunasID)).difference(set([colunaY]))])
            y = np.array(dfDados[colunaY])
        elif self.processType == ProcessContent.SELECT01:
            colunas = set(colunasX1).union(set([colunaY])).difference(set(colunasID))
            dfDados = pd.read_parquet(pathPreprocpq,
                                      columns=colunas,
                                      filters=[
                                          ("DIGITOSISCRICAO", "<=", percent_filter),
                                      ]
                                      )

            X = np.array(dfDados[set(colunasX1).difference(set(colunasID)).difference(set([colunaY]))])
            y = np.array(dfDados[colunaY])
        elif self.processType == ProcessContent.SELECT02:
            colunas = set(colunasX2).union(set([colunaY])).difference(set(colunasID))
            dfDados = pd.read_parquet(pathPreprocpq,
                                      columns=colunas,
                                      filters=[
                                          ("DIGITOSISCRICAO", "<=", percent_filter),
                                      ]
                                      )

            X = np.array(dfDados[set(colunasX2).difference(set(colunasID)).difference(set([colunaY]))])
            y = np.array(dfDados[colunaY])
        else:
            raise NotImplementedError("Not working")


        return X, y

    def TreinoEValidacao_cruzada(self, k, X, y, **kwargs):
        t1 = datetime.datetime.now()

        skf = StratifiedKFold(n_splits=k, random_state=1024, shuffle=True)
        ks = []

        # PROF: para cada parte i de 1 até k
        contador = 1
        metricas_all = []
        for train_index, test_index in skf.split(X, y):
            logger.debug(f"Iteração KFold: {contador}/{k}")
            # print("TRAIN:", train_index, "TEST:", test_index)
            ks.append({"train_index": train_index,
                       "test_index": test_index,
                       "X_train": X[train_index],
                       "X_test": X[test_index],
                       "y_train": y[train_index],
                       "y_test": y[test_index]})

            # Faz o fit (completo para cada Fold i)
            clf = svm.SVC(cache_size=5000, verbose=True, **kwargs)
            clf.fit(X[train_index], y[train_index])

            preditos = clf.predict(X[test_index])

            y_teste = y[test_index]

            metricas = {  # "iteracao": contador,
                "accuracy_score": skmetrics.accuracy_score(y_teste, preditos),
                "f1_score": skmetrics.f1_score(y_teste, preditos),
                "recall_score": skmetrics.recall_score(y_teste, preditos),
                "precision_score": skmetrics.precision_score(y_teste, preditos),
            }
            # print(f"Métricas: \n {metricas}")
            metricas_all = metricas_all + [metricas]

            contador = contador + 1

        tabmetricas = pd.DataFrame(metricas_all)
        logger.info(f"******** RESPOSTAS FINAIS após {str(datetime.datetime.now() - t1)}: \nMétricas atuais: \n")
        # display(tabmetricas)

        logger.info(f"Médias: \n {tabmetricas.mean()}")
        print(scipy.stats.norm.interval(0.98, loc=tabmetricas.mean(), scale=tabmetricas.std()))

        return {"ks": ks,
                "clf": clf,
                "metricas_all": metricas_all,
                "tabmetricas": tabmetricas}

    def processa_plano_id(self, plano_id: int):
        dt_hr_inicio = datetime.datetime.now()
        logger.info("Início")
        k = str(plano_id)

        _ = self.controle_read()
        if k not in self.controle_json.get("experimento", {}).keys():
            raise FileNotFoundError(f"Arquivo não possui plano {plano_id}")

        p = self.controle_json.get("experimento", {}).get(str(plano_id))
        planoX = p.get("parametros", {})
        logger.info(f"**** PlanoAtual: {k}, parâmetros:  {planoX}")
        if p.get('resultados'):
            logger.info(f"Resultados pré-existentes: {p.get('resultados')}")
            return  # Nao precisa fazer nada para esse plano pq já tem resultados

        X, y = self.get_dataset(percent=10, normaliza=False)
        resposta = self.TreinoEValidacao_cruzada(k=10,
                                                 X=np.array(X),
                                                 y=np.array(y),
                                                 **planoX)

        medidas_resposta = {
            "f1_score": resposta['tabmetricas']['f1_score'].mean(),
            "f1_score_ic": scipy.stats.norm.interval(0.98, loc=resposta['tabmetricas']['f1_score'].mean(),
                                                     scale=resposta['tabmetricas']['f1_score'].std()),

            "accuracy_score": resposta['tabmetricas']['accuracy_score'].mean(),
            "accuracy_score_ic": scipy.stats.norm.interval(0.98, loc=resposta['tabmetricas']['accuracy_score'].mean(),
                                                           scale=resposta['tabmetricas']['accuracy_score'].std()),

            "recall_score": resposta['tabmetricas']['recall_score'].mean(),
            "recall_score_ic": scipy.stats.norm.interval(0.98, loc=resposta['tabmetricas']['recall_score'].mean(),
                                                         scale=resposta['tabmetricas']['recall_score'].std()),

            "precision_score": resposta['tabmetricas']['precision_score'].mean(),
            "precision_score_ic": scipy.stats.norm.interval(0.98, loc=resposta['tabmetricas']['precision_score'].mean(),
                                                            scale=resposta['tabmetricas']['precision_score'].std()),
            "Avaliação": 'Total',
            "kernel": planoX.get("kernel"),
            "C": planoX.get("C"),
            "gamma": planoX.get("gamma"),
            "degree": planoX.get("degree"),
            "inicio": dt_hr_inicio.isoformat(),
            "fim": datetime.datetime.now().isoformat(),
        }

        #Pega arquivo atualizado
        arquivo_controle_json = self.controle_read()
        arquivo_controle_json["experimento"][k]['resultados'] = {"medidas_resposta": medidas_resposta,
                                                                 # "metricas_resposta": metricas_resposta
                                                                 }
        self.controle_write(arquivo_controle_json)

        pass




# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    logger.info(f"Parametros: {sys.argv}")
    #Parametro1
    #       ALL = 1
    #       PCA = 2
    #       SELECT01 = 3
    #       SELECT02 = 4
    p1 = int(sys.argv[1])
    #Parametro2 = qual o ID do plano (de 1 a N)
    p2 = int(sys.argv[2])
    mytype = ProcessContent(p1)

    processa = ProcessaSVM_old_version(mytype=mytype)
    processa.processa_plano_id(p2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

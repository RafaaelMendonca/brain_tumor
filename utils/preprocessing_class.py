from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

class CustomMinMaxScaler:
    def __init__(self, feature_range=(0,1)):
        '''
            documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        '''
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, dataFrame):
        try:
            if isinstance(dataFrame, pd.DataFrame) or isinstance(dataFrame, np.ndarray):
                return self.scaler.fit_transform(dataFrame)
            else:
                raise ValueError("O input precisa ser um DataFrame ou um array NumPy.")
        except Exception as e:
            print(f"Erro ao aplicar o MinMaxScaler: {e}")
            raise 
        
class CustomLabelEncoder:
    def __init__(self):
        """
            documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        """
        self.label_encoder = LabelEncoder()

    def fit_transform(self, col):
        try:
            return self.label_encoder.fit_transform(col)
        except Exception as e:
            print(f"Erro ao aplicar o LabelEncoder {e}")
            return None
        
class CustomOneHotEncoder:
    def __init__(self, sparse_output=False, handle_unknown='ignore', drop='if_binary'):
        '''
            documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        '''

    
        self.onehot_encoder = OneHotEncoder(
            sparse_output=sparse_output, #Qaundo True ele retorna uma matrix esparsa no formato "Compressed Sparse Row", False retorna como array denso(mais legível)
            handle_unknown=handle_unknown, #especifica a maneira como categorias desconhecidas são manipuladas durante o Transform
            #ignore' : Quando uma categoria desconhecida é encontrada durante transform, as colunas codificadas one-hot resultantes para esse recurso serão todos zeros. Na transformação inversa, uma categoria desconhecida será indicado como Nenhum.

            drop=drop, #Especifica uma metodologia a ser usada para descartar uma das categorias por característica.
            #if_binary' : descarta a primeira categoria em cada recurso com dois Categorias.
        )

    def fit_transform(self, col):
        try:
            return self.onehot_encoder.fit_transform(col)
        except Exception as e:
            print(f"Erro ao aplicar o OneHotEncoder {e}")

    def get_feature_names_out(self, dataFrame):
        #retorna os nomes das features geradas após a transformação do Onehot Encoder
        try:
            return self.onehot_encoder.get_feature_names_out(dataFrame)
        except Exception as e:
            print(f"Erro ao tentar aplicar o get_feature_names_out {e}")
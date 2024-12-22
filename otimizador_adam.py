import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import mean_absolute_error, r2_score


def msg(mensagem):
    if True:
        print(mensagem)

    return

def plot_grafico_comparacao(comparacoes, diretorio, nome_arquivo):
    
    plt.figure(figsize=(10, 6))

    
    plt.plot(comparacoes['peso_real'], label="Peso Real", marker='o', linestyle='-', alpha=0.7)
    plt.plot(comparacoes['peso_predito'], label="Peso Predito", marker='x', linestyle='--', alpha=0.7)

    
    plt.title("Comparação de Pesos Reais e Preditos", fontsize=16)
    plt.xlabel("Amostra", fontsize=14)
    plt.ylabel("Peso", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    
    caminho_completo = os.path.join(diretorio, nome_arquivo)
    plt.savefig(caminho_completo, format='png', dpi=300)
    print(f"Gráfico salvo em: {caminho_completo}")

    # Mostrar o gráfico
    #plt.show()


def ajustar_brilho_contraste(img, brilho=30, contraste=30):
    img = cv2.convertScaleAbs(img, alpha=1 + contraste / 100.0, beta=brilho)
    return img


def ajusta_imagem(caminho_imagem):

    image = cv2.imread(caminho_imagem)
    if image is None:
        print(f"Erro: Imagem não encontrada ou inválida: {caminho_imagem}")
        return None
    image = ajustar_brilho_contraste(image, brilho=40, contraste=50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    azul_claro = np.array([90, 50, 50])
    azul_escuro = np.array([130, 255, 255])
    
    mascara = cv2.inRange(hsv, azul_claro, azul_escuro)
    mascara_invertida = cv2.bitwise_not(mascara)
    kernel = np.ones((5, 5), np.uint8)
    mascara_invertida_ajustada = cv2.morphologyEx(mascara_invertida, cv2.MORPH_CLOSE, kernel)
    
    resultado = cv2.bitwise_and(image, image, mask=mascara_invertida_ajustada)
    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    
    arquivo_tratado = os.path.basename(caminho_imagem)
    #print( arquivo_tratado )
    cv2.imwrite(f'F:/workspace_cnn/imagens_processadas/{arquivo_tratado}' , resultado)
    return resultado_rgb  


def carrega_arquivos(dataframe, caminho_imagem):
    imagens = []
    peso = []
    print("Carregando arquivos e processando imagens...")
    for _, row in dataframe.iterrows():
        pasta_imagem = os.path.join(caminho_imagem, row['id_tilapia'])
        imagem = cv2.imread(pasta_imagem)
        if imagem is None:
            continue
        imagem_processada = cv2.resize(imagem, (128, 128)) / 255.0
        imagens.append(imagem_processada)
        peso.append(row['peso'])
        
    print("Imagens processadas!")
    return np.array(imagens), np.array(peso)


def carrega_arquivos_processados(dataframe, caminho_imagem):
    imagens = []
    peso = []
    print("Carregando arquivos e processando imagens...")
    for _, row in dataframe.iterrows():
        pasta_imagem = os.path.join(caminho_imagem, row['id_tilapia'])
        imagem_processada = ajusta_imagem(pasta_imagem)
        if imagem_processada is None:
            continue
        imagem_processada = cv2.resize(imagem_processada, (128, 128)) / 255.0
        imagens.append(imagem_processada)
        peso.append(row['peso'])
        
    print("Imagens processadas!")
    return np.array(imagens), np.array(peso)


def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 0.1


def comparar_imagens(csv_path, caminho_imagem, model, epoca, semente):

    print("Iniciando comparação com novas imagens...")
    data = pd.read_csv(csv_path)
    X_test, y_real = carrega_arquivos(data, caminho_imagem)

    y_predito = model.predict(X_test)

    mae = mean_absolute_error(y_real, y_predito)
    r2 = r2_score(y_real, y_predito)

    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Coeficiente de Determinação (R²): {r2}")

    comparacoes = pd.DataFrame({
        "id_tilapia": data['id_tilapia'],
        "peso_real": y_real,
        "peso_predito": y_predito.flatten()
    })

    arquivo = f'comparacoes_{epoca}_seeds_{semente}.csv'
    comparacoes.to_csv(r'F:/workspace_cnn/resultados/'+arquivo, index=False)
    print("Resultados de comparação salvos em 'comparacoes.csv'.")

    arquivo_txt = f'metricas_epoca_{epoca}_seeds_{semente}.csv'
    
    with open(f'F:/workspace_cnn/resultados/{arquivo_txt}', 'w') as f:
        f.write(f"Erro Absoluto Médio (MAE): {mae}\n")
        f.write(f"Coeficiente de Determinação (R²): {r2}\n")
    print(f"Métricas salvas em '{arquivo_txt}'.")

    return comparacoes, mae, r2


#def processa(teste, epoca, semente):
def processa(teste, epoca, semente, com_pre_processamento, learning_rate):

    imagens_treinamento = r'F:/workspace_cnn/treinamento_imagens'
    if teste:
        msg('Rodando teste COMPLETO')
        csv_treinamento = r'F:/workspace_cnn/treinamento_completo.csv'
        csv_comparacao = r'F:/workspace_cnn/teste_completo.csv'
    else:
        msg('Rodando teste PARCIAL')
        csv_treinamento = r'F:/workspace_cnn/treinamento_parcial.csv'
        csv_comparacao = r'F:/workspace_cnn/teste_parcial.csv'

    data = pd.read_csv(csv_treinamento)
    if com_pre_processamento:
        X, y = carrega_arquivos_processados(data, imagens_treinamento)
    else:
        X, y = carrega_arquivos(data, imagens_treinamento)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=90, random_state=42)

    # TENTANDO CRIAR MAIS VARIAÇÕES DO CONJUNTO DE TREINAMENTO
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    #datagen.fit(X_train)
    datagen.fit(X_train, seed=semente)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1, activation='linear')
    ])

    callback = LearningRateScheduler(scheduler)
    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('Compilando modelo...')
    print('-----------------------------------------------')
    print('-----------------------------------------------')
    

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_absolute_error')
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_absolute_error')
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epoca,
        batch_size=32,
        callbacks=[callback]
    )

    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('Avaliando modelo...')
    val_loss = model.evaluate(X_val, y_val)
    print('-----------------------------------------------')
    print('-----------------------------------------------')

    print("Validation Loss (MAE):", val_loss)

    # Comparar novas imagens
    pasta_comparacao = r'F:/workspace_cnn/teste_imagens'
    resultados, mae, r2 = comparar_imagens(csv_comparacao, pasta_comparacao, model, epoca, semente)
    arquivo_grafico = f'grafico_epoca_{epoca}_semente_{semente}.png'
    plot_grafico_comparacao(resultados, r'F:/workspace_cnn/resultados/', arquivo_grafico)
    print('Finalizado ')
    return mae, r2


def main():
    
    resultados = []
    epocas = [ 20, 30, 40, 50, 100 ]
    #epocas = [ 20 ]
    sementes = [ 20, 30, 40, 50, 75, 100 ]
    #learning_rate = 0.1 -- Deu ruim pra caramba
    learning_rate = 0.001
    com_pre_processamento = False
    for epoca in epocas:
        for semente in sementes:
            
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print(f'LR: {learning_rate} EPOCA: {epoca} SEED: {semente}')
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            mae, r2 = processa(True, epoca, semente, com_pre_processamento, learning_rate)
            resultados.append({
                'learning_rate':learning_rate,
                'epoca':epoca,
                'semente': semente,
                'mae': mae,
                'r2': r2
            })
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print(f'FINALIZADOOO EPOCA: {epoca} SEMENTE: {semente}')
            print('-----------------------------------------------')
            print('-----------------------------------------------')
            print('-----------------------------------------------')

        arquivo_resultados = f'F:/workspace_cnn/resultados/adam_lr{learning_rate}_epoca_{epoca}_mae_r2.csv'
        if os.path.exists(arquivo_resultados):
            arquivo_backup = f'F:/workspace_cnn/resultados/adam_lr{learning_rate}_epoca_{epoca}_mae_r2.csv.backup'
            shutil.move(arquivo_resultados,arquivo_backup)

        data_frame_resultados = pd.DataFrame(resultados)
        data_frame_resultados.to_csv(arquivo_resultados, index = False)

    print("Processamento finalizado!")
    

main()
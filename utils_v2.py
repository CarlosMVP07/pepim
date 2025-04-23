import os
import json
import boto3
from botocore.exceptions import NoCredentialsError

def load_face_db(face_db_dir):
    """
    Carica il database dei volti da una directory.
    Ogni file JSON nella directory rappresenta un volto registrato.
    """
    face_db = {}
    if not os.path.exists(face_db_dir):
        os.makedirs(face_db_dir)
        return face_db

    for filename in os.listdir(face_db_dir):
        if filename.endswith(".json"):
            path = os.path.join(face_db_dir, filename)
            with open(path, "r") as f:
                face_data = json.load(f)
                face_db[filename.split(".")[0]] = face_data

    return face_db

def upload_to_s3(local_file, bucket_name, s3_file_path):
    """
    Carica un file locale su un bucket S3.
    
    Args:
        local_file (str): Percorso del file locale.
        bucket_name (str): Nome del bucket S3.
        s3_file_path (str): Percorso del file su S3.
    
    Returns:
        bool: True se il caricamento ha avuto successo, False altrimenti.
    """
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket_name, s3_file_path)
        print(f"File {local_file} caricato con successo su {bucket_name}/{s3_file_path}")
        return True
    except FileNotFoundError:
        print(f"Errore: il file {local_file} non è stato trovato.")
        return False
    except NoCredentialsError:
        print("Errore: credenziali AWS non trovate.")
        return False
    except Exception as e:
        print(f"Errore durante il caricamento su S3: {e}")
        return False
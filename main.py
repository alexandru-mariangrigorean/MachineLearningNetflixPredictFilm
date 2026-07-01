from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Motor de Recomandare Filme")

with open('model_svd_netflix.pkl', 'rb') as file:
    model_svd = pickle.load(file)

df_titles = pd.read_csv(
    'movie_titles.csv', 
    encoding='ISO-8859-1', 
    names=['Movie_ID', 'Year', 'Title'], 
    dtype={'Movie_ID' : str},
    on_bad_lines='skip'                 
)

filme_dict = df_titles.set_index('Movie_ID')[['Year' , 'Title']].to_dict(orient='index')
@app.get("/recomandari/{user_id}")
def obtine_recomandari(user_id: str):
    toate_id_filme = df_titles['Movie_ID'].unique()
    liste_predictii = []

    for movie_id in toate_id_filme:
        pred = model_svd.predict(user_id,movie_id)

        rand_film = df_titles[df_titles['Movie_ID'] == movie_id]

        if not rand_film.empty:
            titlu_real = rand_film['Title'].values[0]
            an_lansare = rand_film['Year'].values[0]
        else:
            titlu_real = 'Titlu necunoscut'
            an_lansare = 'N/A'

        
        liste_predictii.append(
            {
                "movie_id": int(movie_id),
                "titlu": str(titlu_real),
                "an": str(an_lansare),
                "scor_estimat": round(pred.est, 2)
            }
        )
    liste_predictii.sort(key=lambda x : x['scor_estimat'] , reverse=True)
    top_10 = liste_predictii[:10]

    return {
        "user_id": user_id,
        "numar_recomandari": len(top_10),
        "recomandari": top_10
    }    
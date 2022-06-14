
import joblib


def loading_models():
    personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]
    filenames = ["C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/IE.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/NS.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/FT.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/JP.joblib"]
    modelnames = ["IE","NS","FT","JP"]

    for l in range(len(personality_type)):
        filename = filenames[l]
        modelnames[l] = joblib.load(filename)
    return{modelnames}
from flask import Flask, render_template, request
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, hamming
from flask_table import Table, Col

class Results(Table):
    id = Col('Recipe Id', show=False)
    title = Col('Recipe Name')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

def nutrition_hybrid_recommender(recipe_id, sort_order, N):     
    recipe = pd.read_csv('raw-data_recipe.csv')
    df_normalized = pd.read_csv('normalized_nutritions.csv', index_col=0)
    
    allRecipes_cosine = pd.DataFrame(df_normalized.index)
    allRecipes_cosine = allRecipes_cosine[allRecipes_cosine.recipe_id != recipe_id]
    allRecipes_cosine["distance"] = allRecipes_cosine["recipe_id"].apply(lambda x: cosine(df_normalized.loc[recipe_id], df_normalized.loc[x]))
    
    allRecipes_euclidean = pd.DataFrame(df_normalized.index)
    allRecipes_euclidean = allRecipes_euclidean[allRecipes_euclidean.recipe_id != recipe_id]
    allRecipes_euclidean["distance"] = allRecipes_euclidean["recipe_id"].apply(lambda x: euclidean(df_normalized.loc[recipe_id], df_normalized.loc[x]))
    
    allRecipes_hamming = pd.DataFrame(df_normalized.index)
    allRecipes_hamming = allRecipes_hamming[allRecipes_hamming.recipe_id != recipe_id]
    allRecipes_hamming["distance"] = allRecipes_hamming["recipe_id"].apply(lambda x: hamming(df_normalized.loc[recipe_id], df_normalized.loc[x]))
    
    Top10Recommendation_cosine = allRecipes_cosine.sort_values(["distance"]).head(10).sort_values(by=['distance', 'recipe_id'])
    Top10Recommendation_euclidean = allRecipes_euclidean.sort_values(["distance"]).head(10).sort_values(by=['distance', 'recipe_id'])
    Top10Recommendation_hamming = allRecipes_hamming.sort_values(["distance"]).head(10).sort_values(by=['distance', 'recipe_id'])
    
    recipe_df = recipe.set_index('recipe_id')
    hybrid_TopNRecommendation = pd.concat([Top10Recommendation_cosine, Top10Recommendation_euclidean, Top10Recommendation_hamming])
    aver_rate_list = []
    review_nums_list = []
    for recipeid in hybrid_TopNRecommendation.recipe_id:
        aver_rate_list.append(recipe_df.at[recipeid, 'aver_rate'])
        review_nums_list.append(recipe_df.at[recipeid, 'review_nums'])
    hybrid_TopNRecommendation['aver_rate'] = aver_rate_list
    hybrid_TopNRecommendation['review_nums'] = review_nums_list
    TopNRecommendation = hybrid_TopNRecommendation.sort_values(by=[sort_order], ascending=False).head(N)
            
    recipeid_list = list(TopNRecommendation.recipe_id)
    return recipe_df.loc[recipeid_list, :]['recipe_name']
 
@app.route("/recommendation", methods=["POST"])
def recommendations():
    if request.method == 'POST':
        recipe_id = request.form.get("RecipeID")
        recipe_id = int(recipe_id)
        sort_order = request.form.get("Sortby")
        N = request.form.get("TopN")
        N = int(N)
        output = nutrition_hybrid_recommender(recipe_id, sort_order, N)
        table = Results(output)
        table.border = True
        return render_template("recommendation.html", table=table)

if __name__ == '__main__':
    app.run()

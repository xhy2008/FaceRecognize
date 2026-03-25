import pickle
def analysis(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
        same_sims = data["same_sims"]
        diff_sims = data["diff_sims"]
        print(f"file: {file}")
        print(f"min same sim: {min(same_sims)}")
        print(f"avg same sim: {sum(same_sims)/len(same_sims)}")
        print(f"max diff sim: {max(diff_sims)}")
        print(f"avg diff sim: {sum(diff_sims)/len(diff_sims)}")
        print("="*50)

file_list=["lfw_vit_sims.pkl","lfw_sims.pkl","lfw_mobilefacenet_sims.pkl"]
for file in file_list:
    analysis(file)
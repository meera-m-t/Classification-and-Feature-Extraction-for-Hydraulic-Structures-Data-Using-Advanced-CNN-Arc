# # from mnist_dataset import get_mnist
# from gaussian_classifier import gaussian_classifier
# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# def load_data():
#     data = np.load(r"/home/sameerahtalafha/Downloads/5Cdata/data5C.npy") # 5 channels data,(num,100,100,5), channel 0-4 is DEM,RGB,NIR
#     label = np.load(r"/home/sameerahtalafha/Downloads/5Cdata/label.npy")

#     data=data[:,:,:,:1]
#     data=data.reshape(data.shape[0],10000)
#     x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=1,stratify=label)
#     # x_train,x_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=1,stratify=y_train)
#     print(len(y_test))
#     return(x_train, y_train), (x_test, y_test)
# def main():
#     # (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
#     (x_train, y_train), (x_test, y_test) = load_data()
#     # 4.1
#     print("Gaussian classifier with mnist dataset reduced by PCA")
#     pca_components = list(range(1, 20))
#     pca_results = []
#     for pca_component in pca_components:
#         pca = PCA(n_components=pca_component)
#         pca.fit(x_train)
#         x_train_tmp = pca.transform(x_train)
#         x_test_tmp = pca.transform(x_test)
#         gauss = gaussian_classifier()
#         gauss.train(x_train_tmp, y_train)
#         yhat = gauss.predict(x_test_tmp)
#         pca_results.append(np.mean(y_test != yhat) * 100)
#     ##Save plot
#     fig, ax = plt.subplots()
#     # plt.title("pca components vs error rate in gaussian classifier")
#     # plt.plot(pca_components, pca_results, label="Error rate on test set alpha=1.0")
#     # plt.legend(loc="upper left")
#     # plt.xticks(pca_components, rotation=70)
#     # plt.xlabel("PCA Dimensions")
#     # plt.ylabel("Eror rate")
#     # plt.savefig("scripts/results/gc_pca_4-1.png")
#     # 4.2
#     print("Gaussian classifier with mnist dataset reduced by PCA with smoothing")
#     alphas = [0.01, 0.5, 0.9]
#     plt.title("pca components vs error rate in gaussian classifier with smoothing")
#     plt.plot(pca_components, pca_results)
#     for alpha in alphas:
#         pca__smooth_results = []
#         for pca_component in pca_components:
#             pca = PCA(n_components=pca_component)
#             pca.fit(x_train)
#             x_train_tmp = pca.transform(x_train)
#             x_test_tmp = pca.transform(x_test)
#             gauss = gaussian_classifier()
#             gauss.train(x_train_tmp, y_train, alpha=alpha)
#             yhat = gauss.predict(x_test_tmp)
#             pca__smooth_results.append(np.mean(y_test != yhat) * 100)
#         plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on test set alpha={alpha}")
#     print((len(y_test)-sum(abs(yhat-y_test)))/len(y_test))
#     ##Save plot
#     plt.legend(loc="upper left")
#     plt.xlabel("PCA Dimensions")
#     plt.ylabel("Eror rate")
#     plt.xticks(pca_components, rotation=70)
#     plt.savefig("scripts/results/gc_pca_4-2.png")






    # from mnist_dataset import get_mnist
from gaussian_classifier import gaussian_classifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import seaborn as sns
def load_data():
    data = np.load(r"/home/sameerahtalafha/Downloads/5Cdata/data5C.npy") # 5 channels data,(num,100,100,5), channel 0-4 is DEM,RGB,NIR
    label = np.load(r"/home/sameerahtalafha/Downloads/5Cdata/label.npy")

    data=data[:,:,:,:1]
    data=data.reshape(data.shape[0],10000)
    x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.1,random_state=1,stratify=label)
    # x_train,x_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=1,stratify=y_train)
    print(len(y_test))
    return(x_train, y_train), (x_test, y_test)
def main():
    # (x_train, y_train), (x_test, y_test) = get_mnist("data/").load_data()
    (x_train, y_train), (x_test, y_test) = load_data()
    # 4.1
    print("Gaussian classifier with mnist dataset reduced by PCA")
    pca_components = list(range(1, 20))
    pca_results = []
    for pca_component in pca_components:
        pca = PCA(n_components=pca_component)
        pca.fit(x_train)
        x_train_tmp = pca.transform(x_train)
        x_test_tmp = pca.transform(x_test)
        gauss = gaussian_classifier()
        gauss.train(x_train_tmp, y_train)
        yhat = gauss.predict(x_test_tmp)
        pca_results.append(np.mean(y_test != yhat) * 100)
    ##Save plot
    # fig, ax = plt.subplots()
    plt.title("Train data for gaussian classifier")
    # plt.plot(pca_components, pca_results, label="Error rate on test set alpha=1.0")
    # plt.legend(loc="upper left")
    # plt.xticks(pca_components, rotation=70)
    # plt.xlabel("PCA Dimensions")
    # plt.ylabel("Eror rate")
    # plt.savefig("scripts/results/gc_pca_4-1.png")
    # 4.2
    print("Gaussian classifier with mnist dataset reduced by PCA with smoothing")
    alphas = [0.01, 0.5, 0.9]
    # plt.title("pca components vs error rate in gaussian classifier with smoothing")
    # plt.plot(pca_components, pca_results)
    for alpha in alphas:
        pca__smooth_results = []
        for pca_component in pca_components:
            pca = PCA(n_components=pca_component)
            pca.fit(x_train)
            x_train_tmp = pca.transform(x_train)
            x_test_tmp = pca.transform(x_test)
            gauss = gaussian_classifier()
            gauss.train(x_train_tmp, y_train, alpha=alpha)
            yhat_1 = gauss.predict(x_train_tmp)
            yhat = gauss.predict(x_test_tmp)
            pca__smooth_results.append(np.mean(y_test != yhat) * 100)
        # plt.plot(pca_components, pca__smooth_results, marker="o", label=f"Error rate on test set alpha={alpha}")
    print((len(y_test)-sum(abs(yhat-y_test)))/len(y_test))
    ##Save plot
    # x_test_tmp=x_test_tmp[:,:2]
    print((len(y_train)-sum(abs(yhat_1-y_train)))/len(y_train))
    # plt.legend(loc="upper left")
    # plt.xlabel("PCA Dimensions")
    # plt.ylabel("Eror rate")
    # plt.xticks(pca_components, rotation=70)
    # plt.savefig("scripts/results/gc_pca_4-2.png")
    sns_plot = sns.scatterplot(x=x_train_tmp[:,0], y=x_train_tmp[:,1], hue=yhat_1)
    sns_plot.figure.savefig("output.png")
if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     main()

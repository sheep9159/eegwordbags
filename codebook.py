import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import joblib
from feature_extractor import eeglstm, device

if __name__ == '__main__':
    LSTM = eeglstm().to(device)
    LSTM.load_state_dict(torch.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\LSTM.pt'))
    LSTM.eval()

    data = torch.from_numpy(
        np.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\norm_sequence.npy').transpose(
            (0, 2, 1))).float().to(device)

    print(data)

    _, feature = LSTM(data)
    feature = feature.cpu().detach().numpy()
    print('feature.shape: ', feature.shape)

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for n_clusters in range_n_clusters:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(feature) + (n_clusters + 1) * 10])

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature)
        cluster_labels = kmeans.fit_predict(feature)

        joblib.dump(kmeans, fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\聚类效果图\特征\{n_clusters}.model')

        silhouette_avg = silhouette_score(feature, cluster_labels)
        sample_silhouette_values = silhouette_samples(feature, cluster_labels)
        print('silhouette_avg:', silhouette_avg)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\聚类效果图\特征\{n_clusters}')
        plt.clf()

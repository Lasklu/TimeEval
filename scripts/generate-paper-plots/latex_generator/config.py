from pathlib import Path

# result paths
result_root_path = Path("data")
result_file_path = result_root_path /  "paper-plots-results-2.csv"

algo_cites = [
    { "method_image_name": "", "method_name": "AD-LTI", "tex_command": "\\adlti[cite]{}"},
    { "method_image_name": "arima", "method_name": "ARIMA", "tex_command": "\\arima[cite]{}"},
    { "method_image_name": "autoencoder", "method_name": "Autoencoder (AE)", "tex_command": "\\ae[cite]{}"},
    { "method_image_name": "bagel", "method_name": "Bagel", "tex_command": "\\bagel[cite]{}"},
    { "method_image_name": "cblof", "method_name": "CBLOF", "tex_command": "\\cblof[cite]{}"},
    { "method_image_name": "cof", "method_name": "COF", "tex_command": "\\cof[cite]{}"},
    { "method_image_name": "copod", "method_name": "COPOD", "tex_command": "\\copod[cite]{}"},
    { "method_image_name": "dae", "method_name": "DenoisingAutoEncoder (DAE)", "tex_command": "\\dae[cite]{}"},
    { "method_image_name": "dbstream", "method_name": "DBStream", "tex_command": "\\dbstream[cite]{}"},
    { "method_image_name": "deepant", "method_name": "DeepAnT", "tex_command": "\\deepant[cite]{}"},
    { "method_image_name": "deepnap", "method_name": "DeepNAP", "tex_command": "\\deepnap[cite]{}"},
    { "method_image_name": "donut", "method_name": "Donut", "tex_command": "\\donut[cite]{}"},
    { "method_image_name": "dspot", "method_name": "DSPOT", "tex_command": "\\dspot[cite]{}"},
    { "method_image_name": "dwt_mlead", "method_name": "DWT-MLEAD", "tex_command": "\\dwtmlead[cite]{}"},
    { "method_image_name": "eif", "method_name": "Extended Isolation Forest (EIF)", "tex_command": "\\eif[cite]{}"},
    { "method_image_name": "encdec_ad", "method_name": "EncDec-AD", "tex_command": "\\encdecad[cite]{}"},
    { "method_image_name": "ensemble_gi", "method_name": "Ensemble GI", "tex_command": "\\ensemblegi[cite]{}"},
    { "method_image_name": "fast_mcd", "method_name": "Fast-MCD", "tex_command": "\\fastmcd[cite]{}"},
    { "method_image_name": "fft", "method_name": "FFT", "tex_command": "\\fft[cite]{}"},
    { "method_image_name": "generic_rf", "method_name": "Random Forest Regressor (RR)", "tex_command": "\\randomforest[cite]{}"},
    { "method_image_name": "generic_xgb", "method_name": "XGBoosting (RR)", "tex_command": "\\xgboosting[cite]{}"},
    { "method_image_name": "grammarviz3", "method_name": "GrammarViz", "tex_command": "\\grammarviz[cite]{}"},
    { "method_image_name": "hbos", "method_name": "HBOS", "tex_command": "\\hbos[cite]{}"},
    { "method_image_name": "health_esn", "method_name": "HealthESN", "tex_command": "\\healthesn[cite]{}"},
    { "method_image_name": "hif", "method_name": "Hybrid Isolation Forest (HIF)", "tex_command": "\\hif[cite]{}"},
    { "method_image_name": "hotsax", "method_name": "HOT SAX", "tex_command": "\\hotsax[cite]{}"},
    { "method_image_name": "hybrid_knn", "method_name": "Hybrid KNN", "tex_command": "\\hybridknn[cite]{}"},
    { "method_image_name": "if_lof", "method_name": "IF-LOF", "tex_command": "\\iflof[cite]{}"},
    { "method_image_name": "iforest", "method_name": "Isolation Forest (iForest)", "tex_command": "\\iforest[cite]{}"},
    { "method_image_name": "img_embedding_cae", "method_name": "ImageEmbeddingCAE", "tex_command": "\\iecae[cite]{}"},
    { "method_image_name": "kmeans", "method_name": "k-Means", "tex_command": "\\kmeans[cite]{}"},
    { "method_image_name": "knn", "method_name": "KNN", "tex_command": "\\knn[cite]{}"},
    { "method_image_name": "laser_dbn", "method_name": "LaserDBN", "tex_command": "\\laserdbn[cite]{}"},
    { "method_image_name": "left_stampi", "method_name": "Left STAMPi", "tex_command": "\\leftstampi[cite]{}"},
    { "method_image_name": "lof", "method_name": "LOF", "tex_command": "\\lof[cite]{}"},
    { "method_image_name": "lstm_ad", "method_name": "LSTM-AD", "tex_command": "\\lstmad[cite]{}"},
    { "method_image_name": "lstm_vae", "method_name": "LSTM-VAE", "tex_command": "\\lstmvae[cite]{}"},
    { "method_image_name": "median_method", "method_name": "MedianMethod", "tex_command": "\\medianmethod[cite]{}"},
    { "method_image_name": "mscred", "method_name": "MSCRED", "tex_command": "\\mscred[cite]{}"},
    { "method_image_name": "mtad_gat", "method_name": "MTAD-GAT", "tex_command": "\\mtadgat[cite]{}"},
    { "method_image_name": "multi_hmm", "method_name": "MultiHMM", "tex_command": "\\multihmm[cite]{}"},
    { "method_image_name": "norma", "method_name": "NormA", "tex_command": "\\norma[cite]{}"},
    { "method_image_name": "normalizing_flows", "method_name": "Normalizing Flows", "tex_command": "\\normalizingflows[cite]{}"},
    { "method_image_name": "novelty_svr", "method_name": "NoveltySVR", "tex_command": "\\noveltysvr[cite]{}"},
    { "method_image_name": "numenta_htm", "method_name": "NumentaHTM", "tex_command": "\\numentahtm[cite]{}"},
    { "method_image_name": "ocean_wnn", "method_name": "OceanWNN", "tex_command": "\\oceanwnn[cite]{}"},
    { "method_image_name": "omnianomaly", "method_name": "OmniAnomaly", "tex_command": "\\omnianomaly[cite]{}"},
    { "method_image_name": "pcc", "method_name": "PCC", "tex_command": "\\pcc[cite]{}"},
    { "method_image_name": "pci", "method_name": "PCI", "tex_command": "\\pci[cite]{}"},
    { "method_image_name": "phasespace_svm", "method_name": "PhaseSpace-SVM", "tex_command": "\\phasespacesvm[cite]{}"},
    { "method_image_name": "pst", "method_name": "PST", "tex_command": "\\pst[cite]{}"},
    { "method_image_name": "random_black_forest", "method_name": "Random Black Forest (RR)", "tex_command": "\\randomblackforest[cite]{}"},
    { "method_image_name": "robust_pca", "method_name": "RobustPCA", "tex_command": "\\robustpca[cite]{}"},
    { "method_image_name": "s_h_esd", "method_name": "S-H-ESD (Twitter)", "tex_command": "\\shesd[cite]{}"},
    { "method_image_name": "sand", "method_name": "SAND", "tex_command": "\\sand[cite]{}"},
    { "method_image_name": "sarima", "method_name": "SARIMA", "tex_command": "\\sarima[cite]{}"},
    { "method_image_name": "series2graph", "method_name": "Series2Graph", "tex_command": "\\seriestograph[cite]{}"},
    { "method_image_name": "sr", "method_name": "Spectral Residual (SR)", "tex_command": "\\sr[cite]{}"},
    { "method_image_name": "sr_cnn", "method_name": "SR-CNN", "tex_command": "\\srcnn[cite]{}"},
    { "method_image_name": "ssa", "method_name": "SSA", "tex_command": "\\ssa[cite]{}"},
    { "method_image_name": "stamp", "method_name": "STAMP", "tex_command": "\\stamp[cite]{}"},
    { "method_image_name": "stomp", "method_name": "STOMP", "tex_command": "\\stomp[cite]{}"},
    { "method_image_name": "subsequence_fast_mcd", "method_name": "Subsequence Fast-MCD", "tex_command": "\\subsequencefastmcd[cite]{}"},
    { "method_image_name": "subsequence_if", "method_name": "Subsequence IF", "tex_command": "\\subsequenceif[cite]{}"},
    { "method_image_name": "subsequence_lof", "method_name": "Subsequence LOF", "tex_command": "\\subsequencelof[cite]{}"},
    { "method_image_name": "tanogan", "method_name": "TAnoGAN", "tex_command": "\\tanogan[cite]{}"},
    { "method_image_name": "tarzan", "method_name": "TARZAN", "tex_command": "\\tarzan[cite]{}"},
    { "method_image_name": "telemanom", "method_name": "Telemanom", "tex_command": "\\telemanom[cite]{}"},
    { "method_image_name": "torsk", "method_name": "Torsk", "tex_command": "\\torsk[cite]{}"},
    { "method_image_name": "triple_es", "method_name": "Triple ES (Holt-Winter's)", "tex_command": "\\triplees[cite]{}"},
    { "method_image_name": "ts_bitmap", "method_name": "TSBitmap", "tex_command": "\\tsbitmap[cite]{}"},
    { "method_image_name": "valmod", "method_name": "VALMOD", "tex_command": "\\valmod[cite]{}"},
    { "method_image_name": "baseline_random", "method_name": "Random Baseline", "tex_command": "Random Baseline"},
    { "method_image_name": "baseline_normal", "method_name": "Normal Baseline", "tex_command": "Normal Baseline"},
    { "method_image_name": "baseline_increasing", "method_name": "Increasing Baseline", "tex_command": "Increasing Baseline"},
]

algo_cite_lut = {}
for acm in algo_cites:
    try:
        algo_cite_lut[acm["method_name"]] = acm["tex_command"]
    except KeyError:
        pass

# fix some typos:
algo_cite_lut["TAnoGan"] = algo_cite_lut["TAnoGAN"]
algo_cite_lut["Random"] = algo_cite_lut["Random Baseline"]
algo_cite_lut["random"] = algo_cite_lut["Random Baseline"]
algo_cite_lut["normal"] = algo_cite_lut["Normal Baseline"]
algo_cite_lut["increasing"] = algo_cite_lut["Increasing Baseline"]

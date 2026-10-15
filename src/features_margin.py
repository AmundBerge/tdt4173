"""Price-minus-water-value features (see xgboost_price_minus_wv.ipynb)."""
# plant -> reservoir directly upstream, from the connections in data/extended/Tokke_Vinje_topology.yaml
UPSTREAM = {
    "Hogga": "Bandak", "Tokke": "Vinjevatn", "Lio": "Byrtevatn", "Byrte": "Botnedalsvatn",
    "Vinje": "Vaamarvatn", "Haukeli": "Vatjern", "Songa": "Songavatn", "Kjela": "Foersvatn",
    "Vesle Kjela": "Kjelavatn",
}


def add_margin_features(f):
    f = f.copy()
    for plant, res in UPSTREAM.items():
        f[f"margin_{plant}"] = f["price"] - f[f"wv_{res}"]
        f[f"margin_end_{plant}"] = f["price"] - f[f"wv_end_{res}"]
        f[f"margin_end_wk_{plant}"] = f["price_week_mean"] - f[f"wv_end_{res}"]
    return f

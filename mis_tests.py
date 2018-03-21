features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features = ['right-x', 'right-y', 'left-x', 'left-y']
for (feature_norm, feature) in zip(features_norm, features):
    print ("feature_norm %s feature %s" %(feature_norm, feature))
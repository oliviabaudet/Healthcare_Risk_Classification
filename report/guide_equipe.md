# Guide d'équipe pour l'oral (3 membres, 14 diapositives)

Projet : prédiction de la réadmission à 30 jours de patients diabétiques (UCI Diabetes 130-US hospitals, 1999-2008). Projet pédagogique, pas un outil médical.

Table des matières : 0 Vue d'ensemble · 1A Membre A · 1B Membre B · 1C Membre C · 2 À savoir par TOUS · 3 Répétition · Sources

---

## 0. L'histoire en bref

1. **Question** : les informations d'un séjour hospitalier d'un patient diabétique permettent-elles de prédire une réadmission dans les 30 jours après la sortie ?
2. **Formulation** : classification binaire supervisée ; cible = 1 si `readmitted` = `<30`, 0 sinon (`>30` ou `NO`).
3. **Données** : 101 766 séjours, 71 518 patients, 11,16 % de réadmissions < 30 j (cible déséquilibrée). Après exclusion de 2 423 décès/hospices : 99 343 séjours (79 541 train, 19 802 test).
4. **Méthode** : prétraitement dans des Pipelines, découpage train/test groupé par patient, validation croisée `GroupKFold` à 5 plis, 5 modèles (régression logistique, arbre, forêt, KNN, HistGradientBoosting), réglage `RandomizedSearchCV`, seuil choisi sur le train.
5. **Résultat** : ROC-AUC ≈ 0,67 pour HGB, forêt et régression logistique ; F1 ≈ 0,28 ; précision ≈ 0,2 pour un rappel ≈ 0,5.
6. **Pas de gagnant net** : les écarts entre ces trois modèles sont du même ordre que la variabilité de la CV (écart-type ≈ 0,010) ; un seul split de test.
7. **Réglage** : gain modeste et inégal (pour HGB, aucun gain sur le test).
8. **Variable la plus nette** : `number_inpatient` (hospitalisations antérieures), r = 0,165 ; association, jamais causalité.
9. **Clustering** (K-Means k=3 sur 8 variables standardisées, PCA pour la visualisation, train et test confondus) : silhouette 0,187, clusters faiblement séparés, d'où le choix du supervisé.
10. **Message final** : modèle modeste, trie mieux que le hasard mais trop imprécis pour décider seul ; au mieux un outil de repérage (screening), jamais un outil clinique sans validation.

### Répartition de la parole (≈ 3 minutes chacun)

| Membre | Slides | Contenu |
|---|---|---|
| A | 1 à 4 (cover, objectif, données, exploration) | Contexte, classification vs clustering, données, exclusion décès/hospice, exploration |
| B | 5 à 9 (protocole, modèles, résultats, validation croisée, courbes) | Prétraitement, split groupé, GroupKFold, 5 modèles, hors-cours, métriques, résultats test et CV |
| C | 10 à 14 (réglage, variables, analyse, limites, suites) | Tuning et seuil, importance des variables, discussion, limites, recommandations |

Les parties sont échangeables : tout le monde doit connaître la section 2 et les chiffres de la section 2.3, et savoir répondre aux questions des deux autres membres.

---

## 1A. Membre A : slides 1 à 4

### (a) Ce que tu présentes

**Slide 1, cover** : « Predicting 30-day readmission of diabetic patients », classification supervisée sur 99 343 séjours d'hôpitaux américains, 1999-2008. Dis d'emblée : projet pédagogique pour le cours Introduction to Machine Learning, jeu public UCI, ce n'est pas un outil médical validé. Présente l'équipe (Olivia Baudet, Paolo, Garance).

**Slide 2, objectif (Research question)** : la question est « les informations d'un séjour permettent-elles de prédire une réadmission dans les 30 jours ? ». On la traduit en classification binaire supervisée : la cible vaut 1 si le patient revient dans les 30 jours, 0 sinon. On a aussi testé une approche non supervisée (clustering) : K-Means (k=3) sur les 8 variables numériques standardisées ; la PCA sert à mesurer la variance et à visualiser les clusters en 2D. Silhouette (calculée sur 10 000 séjours) de 0,187, donc des groupes mal séparés, même si le taux de réadmission varie de 9,2 % à 17,1 % selon le cluster. Comme on dispose d'une étiquette fiable (on sait qui est revenu), l'apprentissage supervisé est mieux adapté ; le clustering reste un outil de description.

**Slide 3, données (Data)** : le fichier contient 101 766 séjours de 71 518 patients uniques. Un patient peut avoir plusieurs séjours, donc les lignes ne sont pas indépendantes (16 773 patients ont plus d'un séjour). 11,16 % des séjours sont suivis d'une réadmission < 30 jours : la cible est déséquilibrée. On retire 2 423 séjours terminés par un décès ou un transfert en hospice (codes de sortie 11, 13, 14, 19, 20, 21), car une réadmission y est impossible ou sans sens. Il reste 99 343 séjours, séparés en 79 541 pour l'entraînement et 19 802 pour le test. Plusieurs variables sont presque vides : `weight` 96,86 % de manquants, `max_glu_serum` 94,75 %, `A1Cresult` 83,28 %, `medical_specialty` 49,08 %, `payer_code` 39,56 %.

**Slide 4, exploration (Weak signals)** : le taux de réadmission varie fortement avec le mode de sortie (`discharge_disposition_id`) : code 1 (domicile) 9,30 %, code 3 14,66 %, code 2 16,07 %, code 5 20,86 %, code 22 27,70 %, contre 11,16 % en moyenne. Côté variables numériques, la plus corrélée à la cible est `number_inpatient` (nombre d'hospitalisations antérieures) avec r = 0,165 ; toutes les autres sont à 0,061 ou moins. Conclusion de la slide : aucune variable numérique n'est fortement liée à la cible, donc on s'attend à des performances modestes. Précise que l'exploration porte sur le fichier complet (101 766 lignes), y compris les séjours exclus ensuite (le code 11 y apparaît avec 0,00 %).

### (b) Ce que tu dois absolument savoir

| Chiffre | Valeur | Source |
|---|---|---|
| Séjours / patients | 101 766 / 71 518 | `eda_summary.txt` |
| Patients avec plus d'un séjour | 16 773 | `eda_summary.txt` |
| Réadmissions < 30 j | 11 357 = 11,16 % | `eda_summary.txt` |
| Exclus (décès/hospice) | 2 423 | `README.md`, rapport |
| Conservés | 99 343 | rapport |
| Train / test | 79 541 / 19 802 | rapport (comptage `y_train.csv`, `y_test.csv`) |
| Positifs train / test | 9 098 (0,114) / 2 216 (0,112) | rapport |
| Corrélation `number_inpatient` | 0,165 | `eda_summary.txt` |
| Silhouette k=3 | 0,187 (k=2 : 0,182) | `clustering_summary.txt` |
| Variance PCA | PC1 25,5 % ; PC1+PC2 43,6 % ; 5 composantes 77,3 % | `clustering_summary.txt` |
| Taux par cluster | 0,092 (n=56 019) ; 0,171 (n=14 140) ; 0,128 (n=29 184) ; global 0,114 | `clustering_summary.txt` |

**Décisions de méthode et justifications**
- **Cible binaire** : `<30` = 1, `>30` et `NO` = 0. La classe positive est la réadmission rapide.
- **Exclusion des codes 11/13/14/19/20/21** : décès ou hospice. Un patient décédé ne peut pas être réadmis ; les garder fausserait la cible (le code 11 a 0,00 % de réadmission dans l'EDA).
- **Pas de `encounter_id` ni `patient_nbr` comme variables** : ce sont des identifiants. `patient_nbr` sert seulement à regrouper les séjours d'un même patient.
- **Pourquoi supervisé plutôt que clustering** : étiquette disponible, silhouette faible (0,187 < 0,25).
- **Clustering = exploratoire** (K-Means k=3 sur 8 variables standardisées, PCA pour la visualisation, train et test confondus) : fait sur la population filtrée (99 343 séjours, `log1p` sur 4 variables de comptage puis `StandardScaler`, silhouette sur 10 000 séjours) ; il n'alimente pas les classifieurs.
- Différence de taux : 11,16 % (fichier complet) contre 11,39 % (population filtrée, clustering). Les deux sont corrects, populations différentes.

### (c) Questions probables du jury

1. **Pourquoi avoir exclu les décès et hospices ?** Une réadmission y est impossible ou non pertinente ; les inclure fausserait la cible. 2 423 séjours retirés sur 101 766.
2. **Pourquoi classification et pas clustering ?** Nous avons une étiquette fiable. Le clustering (K-Means k=3) donne une silhouette de 0,187, donc des clusters mal séparés ; il contient un signal (taux 9,2 % à 17,1 %) mais ne sépare pas nettement les patients.
3. **Pourquoi k=3 ?** C'est le maximum de silhouette (0,187), mais très proche de k=2 (0,182) ; la différence n'est pas décisive. Le choix reste exploratoire.
4. **Les clusters prouvent-ils des profils de patients à risque ?** Non. Les taux par cluster sont descriptifs, sans intervalle de confiance, et les séjours ne sont pas indépendants. Le rapport max/min de 1,85 est une observation.
5. **La cible est déséquilibrée, est-ce un problème ?** Oui : 11,16 % de positifs. L'accuracy devient trompeuse (toujours prédire 0 donne 0,888 sur le test) ; on utilise ROC-AUC, PR-AUC, rappel, précision, F1 et `class_weight`.
6. **Les patients apparaissent plusieurs fois, quelle conséquence ?** Les séjours ne sont pas indépendants. D'où le split et la CV groupés par patient (détaillé par le membre B).
7. **Beaucoup de valeurs manquantes, qu'en faites-vous ?** Imputation simple (médiane ou valeur la plus fréquente) ; limite assumée. `weight`, `max_glu_serum`, `A1Cresult` sont presque vides.
8. **Pourquoi la PCA n'a-t-elle pas simplifié davantage ?** La variance est répartie (25,5 ; 18,2 ; 12,6 %...) ; il faut 5 composantes pour 77,3 % : les 8 variables apportent des informations assez distinctes.
9. **Pourquoi peu de signal dans les variables ?** Les corrélations sont faibles (max 0,165) ; on s'attendait donc à des performances modestes, ce que confirment les résultats.
10. **Ce modèle est-il utilisable à l'hôpital ?** Non, pas en l'état (voir membre C).

### (d) Pièges : ce qu'il ne faut PAS dire

- Ne pas dire que les clusters « identifient les patients à risque » ou « séparent bien » : la silhouette est faible.
- Ne pas dire que `number_inpatient` « cause » les réadmissions : association seulement.
- Ne pas dire « 11 % » sans préciser la population (fichier complet 11,16 % ; filtrée 11,39 %).
- Ne pas présenter les taux des tranches d'âge jeunes comme des résultats solides : [0-10) n=161, [10-20) n=691, petits effectifs.
- Ne pas dire que le projet est « un outil médical » ni laisser entendre une utilité clinique.
- Ne pas dire que l'exclusion « améliore » le modèle : elle sert à rendre la cible cohérente.

### (e) Transition vers B

« Voilà les données et ce qu'elles nous disent : des signaux faibles et une cible déséquilibrée. Mon collègue va maintenant expliquer comment nous avons construit un protocole d'évaluation honnête, et ce que donnent les cinq modèles. »

---

## 1B. Membre B : slides 5 à 9

### (a) Ce que tu présentes

**Slide 5, protocole (Preprocessing and protocol)** : à gauche le prétraitement. Découpage 80/20 groupé par patient (`GroupShuffleSplit`, `random_state=42`, `patient_nbr` comme groupe) : un patient a souvent plusieurs séjours très semblables ; s'ils étaient répartis entre train et test, le modèle pourrait « reconnaître » le patient et le score serait gonflé (fuite de données, data leakage). Variables numériques : imputation par la médiane puis standardisation. Variables catégorielles : valeur la plus fréquente puis one-hot (y compris les codes `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, qui sont des catégories et non des mesures). Tout est ajusté sur le train seul, dans des `Pipeline`. À droite l'évaluation : CV `GroupKFold` à 5 plis sur le train ; réglage par `RandomizedSearchCV` (au plus 8 tirages, grille complète pour la régression logistique, 5, et le KNN, 6 ; score PR-AUC ; sous-échantillon de 26 514 lignes, KNN : recherche sur ses 20 000 lignes) ; seuil choisi sur les prédictions hors-pli (out-of-fold) du train en maximisant le F1. Le test n'a jamais servi à choisir un modèle, un réglage ou un seuil.

**Slide 6, modèles (Five models compared)** : cinq modèles et leur configuration de base : régression logistique (C=1, `max_iter=1000`), arbre de décision (`max_depth=12`), forêt aléatoire (100 arbres, `max_depth=18`), HistGradientBoosting (paramètres par défaut, sans arrêt précoce, `early_stopping=False`), KNN (k=5). `class_weight="balanced"` pour la régression logistique, l'arbre, la forêt et HGB ; le KNN de scikit-learn ne le supporte pas. Vu au cours : `StandardScaler`, K-Means, PCA, silhouette, découpage train/test, KNN mais en régression seulement. Hors cours (expliqué dans le rapport) : régression logistique, arbre, forêt, HGB, métriques de classification, ROC-AUC et PR-AUC, `GroupKFold`, `RandomizedSearchCV`, seuil de décision. La consigne du projet encourage les techniques hors cours.

**Slide 7, résultats (Baseline models on the test set)** : tableau sur le test à seuil 0,5. HGB mène : F1 0,275, ROC-AUC 0,670, PR-AUC 0,227 ; forêt (F1 0,271, ROC-AUC 0,662, PR-AUC 0,197) et régression logistique (0,266 ; 0,656 ; 0,213) sont très proches. L'arbre est derrière (0,246 ; 0,625 ; 0,185). Le KNN a la meilleure accuracy (0,879) mais un rappel de 0,045 : il prédit presque toujours 0. Un modèle aléatoire aurait ROC-AUC 0,5 et PR-AUC ≈ 0,11 (la prévalence). Test = un seul split sans intervalle de confiance : ces petits écarts n'établissent pas un classement.

**Slide 8, validation croisée (Grouped cross-validation)** : 5 plis `GroupKFold` sur le train (79 541 lignes), tous les séjours d'un patient dans le même pli. HGB en tête : F1 0,279 ± 0,009, ROC-AUC 0,669 ± 0,010. Pour HGB, forêt et régression logistique, les barres d'erreur se recouvrent, sauf la ROC-AUC de la régression logistique (0,647) face à HGB (et le rappel forêt contre HGB). Il s'agit de la CV des modèles de base (configuration de 03), pas des modèles réglés. L'écart-type entre plis n'est pas un test statistique. Le KNN est nettement en retrait mais n'est évalué que sur 20 000 lignes, donc non comparable à données égales.

**Slide 9, courbes (ROC and precision-recall curves)** : lignes pleines = modèles de base, pointillés = réglés. Les courbes de la régression logistique, de la forêt et de HGB sont presque superposées (ROC-AUC ≈ 0,67), nettement au-dessus de la diagonale mais loin de la courbe idéale : le modèle classe mieux que le hasard sans être précis. Sur la courbe précision-rappel, la précision retombe vers la prévalence (≈ 0,11) quand le rappel augmente et ne dépasse 0,4 que pour de très faibles rappels. L'arbre et le KNN sont en dessous.

### (b) Ce que tu dois absolument savoir

**Tableau test, base (`model_comparison.csv`)**

| Modèle | Acc. | Précision | Rappel | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| HGB | 0,662 | 0,181 | 0,571 | 0,275 | 0,670 | 0,227 |
| Random Forest | 0,668 | 0,180 | 0,552 | 0,271 | 0,662 | 0,197 |
| Logistic Regression | 0,659 | 0,175 | 0,554 | 0,266 | 0,656 | 0,213 |
| Decision Tree | 0,592 | 0,155 | 0,596 | 0,246 | 0,625 | 0,185 |
| KNN (k=5) | 0,879 | 0,265 | 0,045 | 0,076 | 0,563 | 0,137 |

**Tableau CV, 5 plis (`cv_results.csv`), moyenne ± écart-type**

| Modèle | Lignes | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 79 541 | 0,266 ± 0,007 | 0,647 ± 0,010 | 0,208 ± 0,013 |
| Decision Tree | 79 541 | 0,255 ± 0,005 | 0,622 ± 0,007 | 0,182 ± 0,006 |
| Random Forest | 79 541 | 0,272 ± 0,011 | 0,662 ± 0,011 | 0,200 ± 0,012 |
| KNN | 20 000 | 0,063 ± 0,010 | 0,562 ± 0,008 | 0,143 ± 0,005 |
| HGB | 79 541 | 0,279 ± 0,009 | 0,669 ± 0,010 | 0,224 ± 0,014 |

**Décisions de méthode et justifications**
- **Split par patient** : évite que les séjours d'un même patient soient dans train et test (fuite de données, score gonflé).
- **`GroupKFold`** : même logique pour la CV ; sinon les plis partageraient des patients.
- **Tout ajusté sur le train dans des Pipelines** : médianes, valeurs fréquentes, moyennes de standardisation et colonnes one-hot viennent du train uniquement, pas du test.
- **`class_weight="balanced"`** : pondère davantage les erreurs sur la classe rare (11 %), sinon les modèles prédisent presque toujours 0. Le KNN ne le supporte pas, d'où son rappel très bas.
- **Pourquoi l'accuracy est trompeuse** : 2 216 positifs sur 19 802 ; « toujours 0 » donne 0,888 sans détecter aucun cas. Le KNN réglé à 0,5 atteint 0,888 avec un rappel de 0,009.
- **Métriques retenues** : ROC-AUC, PR-AUC, rappel, précision, F1 de la classe 1.
- **KNN sur 20 000 lignes** pour la CV (coût de calcul, environ 2 400 colonnes après one-hot) ; la ligne KNN de `cv_results.csv` est le KNN de base sur ces 20 000 lignes. Le KNN du tableau test de base est, lui, entraîné sur les 79 541 lignes.
- **Modèles hors cours** : voir le tableau section 2.2.

### (c) Questions probables du jury

1. **Pourquoi l'accuracy est-elle trompeuse ici ?** 88,8 % des séjours du test sont négatifs : prédire toujours 0 donne 0,888 d'accuracy. Le KNN a 0,879 mais un rappel de 0,045.
2. **Pourquoi couper train/test par patient ?** Les séjours d'un patient se ressemblent ; s'ils sont des deux côtés, le modèle peut reconnaître le patient et le score est gonflé (fuite de données).
3. **Pourquoi utiliser des modèles hors cours ?** La consigne du projet encourage les techniques hors cours ; nous les avons utilisées et expliquées dans le rapport. Les notions vues au cours (scaler, PCA, K-Means, silhouette, split) sont aussi utilisées.
4. **Quel est le meilleur modèle ?** Pas de gagnant net. HGB, forêt et régression logistique forment un groupe à ROC-AUC 0,66-0,67 ; les écarts sont du même ordre que l'écart-type de la CV (≈ 0,010) ; test = un seul split.
5. **HGB est pourtant premier partout, non ?** En base, oui sur F1, ROC-AUC et PR-AUC (test et CV). Mais l'écart avec la forêt est inférieur à l'écart-type (0,669 ± 0,010 contre 0,662 ± 0,011). HGB devance la régression logistique de base d'environ deux écarts-types en ROC-AUC de CV (0,669 contre 0,647), mais sur le test, une fois les deux réglés, l'écart disparaît (régression logistique réglée 0,670 contre HGB réglé 0,668).
6. **Pourquoi le KNN est-il si mauvais ?** Rappel de 0,045 : il prédit presque toujours 0 (pas de `class_weight` possible). Il est aussi évalué sur 20 000 lignes en CV : la comparaison n'est pas équitable et ne prouve pas que la méthode est moins adaptée à données égales.
7. **Qu'est-ce que la ROC-AUC de 0,67 veut dire ?** Un cas positif reçoit un score supérieur à un cas négatif dans environ 67 % des paires (0,5 = hasard). C'est mieux que le hasard, mais modeste.
8. **Pourquoi PR-AUC en plus de ROC-AUC ?** Plus informative quand la classe positive est rare ; la référence aléatoire est la prévalence (≈ 0,11).
9. **Le test a-t-il été utilisé pour choisir quelque chose ?** Non : ni modèle, ni hyperparamètre, ni seuil. Le seuil est choisi sur les prédictions hors-pli du train.
10. **La CV donne-t-elle des intervalles de confiance ?** Non : l'écart-type entre plis donne un ordre de grandeur, ce n'est pas un test statistique.
11. **Pourquoi one-hot pour `discharge_disposition_id` alors que c'est un nombre ?** Ce sont des codes catégoriels, pas des quantités ; l'ordre numérique n'a pas de sens.

### (d) Pièges : ce qu'il ne faut PAS dire

- Ne pas dire « HGB est le meilleur modèle » sans nuance ; dire « légèrement en tête, sans écart démontré avec la forêt et la régression logistique ».
- Ne pas citer l'accuracy du KNN (0,879) comme une qualité.
- Ne pas dire que l'écart-type de CV « prouve » ou « réfute » une différence : ce n'est pas un test.
- Ne pas dire que le KNN est « la pire méthode » : il est entraîné sur moins de données.
- Ne pas dire « ROC-AUC 0,67, c'est bon » : c'est modeste.
- Ne pas confondre : les tableaux de base sont à seuil 0,5, sans réglage (le réglage est présenté par C).
- Ne pas dire que la CV prouve que le modèle généralisera : un seul split de test, données de 1999-2008.

### (e) Transition vers C

« Voilà les résultats des modèles de base : trois modèles au coude à coude autour de 0,67 de ROC-AUC. Ma collègue/mon collègue va montrer ce que changent le réglage des hyperparamètres et le choix du seuil, puis ce que les résultats permettent, ou non, de conclure. »

---

## 1C. Membre C : slides 10 à 14

### (a) Ce que tu présentes

**Slide 10, réglage (Effect of tuning and threshold)** : le tableau compare base et réglé. ROC-AUC base → réglé : régression logistique 0,656 → 0,670 ; forêt 0,662 → 0,671 ; HGB 0,670 → 0,668 ; arbre 0,625 → 0,650 ; KNN 0,563 → 0,624. Ensuite l'effet du seuil (0,5 → seuil optimisé) : le rappel baisse et la précision monte, le F1 bouge à peine (LR 0,276 → 0,278 ; forêt 0,277 → 0,283 ; HGB 0,268 → 0,277 ; arbre 0,260 → 0,265). Explique : le seuil ne change pas le classement des patients (AUC identique) mais déplace le point de fonctionnement : combien d'alertes on accepte. Pour le KNN, le réglé à 0,5 ne produit que 40 alertes sur 19 802 (rappel 0,009), son seuil optimisé est bas (0,117) et donne rappel 0,535, F1 0,245. Le KNN est réglé sur 20 000 lignes seulement.

**Slide 11, variables (Feature importance)** : pour la forêt aléatoire, `number_inpatient` domine (≈ 0,13), loin devant `discharge_disposition_id_1` (≈ 0,04) ; c'est aussi la variable la plus corrélée à la cible (r = 0,165), cohérent avec l'exploration. Attention : les coefficients extrêmes de la régression logistique concernent des codes de diagnostic très rares (les plus forts coefficients positifs portent sur des codes très rares, 3 à 6 séjours, par exemple `diag_1_643` : 4 ; les négatifs affichés sur 26 à 48 séjours) : instables, ce ne sont pas des facteurs de risque. Association, jamais causalité.

**Slide 12, analyse (Reading the results with caution)** : quatre messages. 0,888 : accuracy trompeuse (toujours 0), le KNN réglé à 0,5 l'atteint avec un rappel de 0,009. ≈ 0,67 : pas de gagnant net entre HGB réglé, forêt et régression logistique ; écarts du même ordre que l'écart-type de CV (≈ 0,010). ≈ 0,2 : précision modeste ; à un rappel d'environ 0,5, environ 4 alertes sur 5 sont de fausses alertes. Le modèle trie mieux que le hasard mais n'est pas assez précis pour décider seul. Quatrième encadré : le KNN prédit presque toujours 0 (rappel 0,045) ; la version réglée n'utilise que 20 000 lignes, la comparaison n'est pas équitable. Un seul split de test, pas d'intervalle de confiance.

**Slide 13, limites (Limitations)** : méthode : un seul split de test sans intervalle de confiance ; réglage limité (26 514 lignes, au plus 8 tirages, petites grilles) ; KNN sur 20 000 lignes. Données et usage : hôpitaux américains 1999-2008 ; certaines variables ne sont connues qu'à la sortie ; décès et hospices exclus ; pas de calibration des probabilités ; pas d'analyse d'équité par sous-groupes ; sans validation ni revue d'experts, pas de décision clinique.

**Slide 14, suites (Recommendations and next steps)** : usage au mieux un outil de repérage (screening) pour prioriser le suivi, jamais une décision automatique. Le seuil se règle selon la capacité de suivi : seuil bas = plus de rappel mais plus de fausses alertes. Signal le plus net : hospitalisations antérieures (association, pas causalité). Pistes non testées : nouvelles variables (comorbidités, diagnostics regroupés, historique du patient), données plus récentes, plusieurs découpages. La slide liste grilles plus larges sur tout le train, calibration, XGBoost/LightGBM, analyse d'équité ; le rapport ajoute SMOTE. Tout cela est non testé.

### (b) Ce que tu dois absolument savoir

**Hyperparamètres retenus (`best_params.json`)** : LR C=0,01 ; arbre `max_depth`=8, `min_samples_leaf`=200 ; forêt 200 arbres, `max_depth`=18, `min_samples_leaf`=5, `max_features`=0,1 ; KNN k=101, `weights`=distance ; HGB `learning_rate`=0,05, `max_iter`=100, `max_leaf_nodes`=15, `l2_regularization`=10,0.

**Seuils optimisés (`thresholds.json`)** : LR 0,521 ; arbre 0,549 ; forêt 0,516 ; KNN 0,117 ; HGB 0,535.

**Test, modèles réglés (`tuned_comparison.csv`)**

| Modèle | Seuil | Précision | Rappel | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LR | 0,5 | 0,183 | 0,559 | 0,276 | 0,670 | 0,217 |
| LR | 0,521 | 0,192 | 0,503 | 0,278 | 0,670 | 0,217 |
| Arbre | 0,5 | 0,166 | 0,597 | 0,260 | 0,650 | 0,203 |
| Arbre | 0,549 | 0,193 | 0,426 | 0,265 | 0,650 | 0,203 |
| Forêt | 0,5 | 0,187 | 0,530 | 0,277 | 0,671 | 0,226 |
| Forêt | 0,516 | 0,200 | 0,482 | 0,283 | 0,671 | 0,226 |
| HGB | 0,5 | 0,174 | 0,587 | 0,268 | 0,668 | 0,223 |
| HGB | 0,535 | 0,194 | 0,484 | 0,277 | 0,668 | 0,223 |
| KNN (20 000) | 0,5 | 0,525 | 0,009 | 0,019 | 0,624 | 0,190 |
| KNN (20 000) | 0,117 | 0,159 | 0,535 | 0,245 | 0,624 | 0,190 |

**Décisions de méthode et justifications**
- **Réglage par PR-AUC** (`average_precision`) : métrique adaptée à une classe rare et indépendante du seuil.
- **Réglage sur un sous-échantillon** (26 514 lignes, un tiers du train, stratifié et groupé) avec `RandomizedSearchCV` (`n_iter` = min(8, taille de la grille) : LR 5 tirages, KNN 6, soit la grille complète ; KNN : recherche sur ses 20 000 lignes) : limité par le temps de calcul (environ 10 minutes pour `07`, 22 cœurs) ; puis réentraînement sur tout le train (sauf KNN, 20 000 lignes). Ce n'est pas une optimisation poussée.
- **Seuil choisi sur le train (prédictions hors-pli), maximisant le F1** : le test ne doit jamais servir à choisir ; sinon le score test serait optimiste. Limite : ces prédictions viennent de modèles entraînés sur 4/5 du train, le seuil final peut être légèrement décalé.
- **Effet du seuil** : le F1 gagne entre 0,002 et 0,009 pour LR, arbre, forêt, HGB ; le vrai changement est le compromis rappel/précision.
- **Effet du réglage** : modeste et inégal. Gains de ROC-AUC de LR, forêt et HGB ≤ 0,014, du même ordre que la variabilité de la CV ; arbre +0,025 ROC-AUC et forêt ≈ +0,03 PR-AUC plus nets mais sur un seul test. Pour HGB, pas mieux que la base (0,670 → 0,668).
- **Pourquoi les décisions sont « modestes »** : ROC-AUC ≈ 0,67 ; précision ≈ 0,2.
- **Variables disponibles à la sortie** : `discharge_disposition_id` et le nombre de diagnostics ne sont connus qu'à la sortie ; le modèle ne conviendrait qu'à une prédiction faite à la sortie. L'analyse de chaque variable n'a pas été faite dans le projet.
- **Importance de la forêt par impureté** : biaisée en faveur des variables numériques ou à nombreuses valeurs ; valeurs lues sur le graphique (non sauvegardées).

### (c) Questions probables du jury

1. **Le réglage améliore-t-il les modèles ?** Un peu et de façon inégale : LR 0,656 → 0,670, forêt 0,662 → 0,671, arbre 0,625 → 0,650 en ROC-AUC, mais HGB 0,670 → 0,668. Les gains de LR, forêt, HGB sont du même ordre que la variabilité de la CV.
2. **Pourquoi choisir le seuil sur le train et pas sur le test ?** Pour que le test reste indépendant ; choisir sur le test rendrait le score optimiste. On utilise les prédictions hors-pli du train.
3. **Le seuil optimisé améliore-t-il le modèle ?** Il ne change pas le classement (AUC identique) et le F1 bouge de 0,002 à 0,009. Il déplace le compromis : moins de rappel, plus de précision.
4. **Pourquoi `class_weight` ?** Compenser la rareté de la classe positive (11 %). Effet secondaire : les probabilités ne sont pas calibrées (moyenne prédite ≈ 0,42 à 0,47 selon le modèle sur le test pour une prévalence de 0,11 ; base : arbre 0,420, HGB 0,446, LR 0,448, forêt 0,469 ; réglés : forêt 0,438, arbre 0,461, LR 0,463, HGB 0,463).
5. **Pourquoi pas de gagnant net ?** HGB, forêt et LR ont des ROC-AUC de 0,66-0,67 avec des écarts du même ordre que l'écart-type de CV (≈ 0,010) ; un seul split de test, sans intervalle de confiance. Classement changeant selon la métrique pour les modèles réglés (forêt réglée : meilleure PR-AUC 0,226 et meilleur F1 0,283 au seuil optimisé).
6. **Ce modèle est-il utilisable en clinique ?** Non. Scores modestes, environ 4 alertes sur 5 fausses à un rappel ≈ 0,5, pas de calibration, pas d'analyse d'équité, données de 1999-2008, un seul split. Au mieux outil de repérage, après validation, gouvernance et revue d'experts.
7. **`number_inpatient` est-elle une cause de réadmission ?** Non : association. Les résultats ne permettent pas de conclusion causale.
8. **Que montrent les coefficients de la régression logistique ?** Rien de fiable : les plus forts coefficients positifs portent sur des codes très rares (3 à 6 séjours, par exemple `diag_1_643` : 4), les négatifs affichés sur 26 à 48 séjours, donc instables ; et ils viennent du modèle de base (C=1), pas du réglé (C=0,01).
9. **Que feriez-vous pour améliorer ?** Pistes non testées : regrouper les codes diagnostics, ajouter des variables d'historique, grilles plus larges sur tout le train, calibration, plusieurs découpages, données plus récentes.
10. **Le KNN réglé semble avoir un F1 de 0,019 : est-ce un bug ?** Non : à 0,5, seules 40 probabilités sur 19 802 dépassent le seuil. Au seuil optimisé (0,117), F1 = 0,245 et rappel = 0,535.
11. **Toutes les variables sont-elles connues à la date de prédiction ?** Non : `discharge_disposition_id` et le nombre de diagnostics sont connus à la sortie. Le modèle n'a de sens qu'à la sortie ; l'analyse variable par variable n'a pas été menée.

### (d) Pièges : ce qu'il ne faut PAS dire

- Ne pas dire que le réglage « améliore nettement » : gains modestes, nuls pour HGB.
- Ne pas dire que le seuil optimisé « améliore le modèle » : il déplace le compromis.
- Ne pas dire « le modèle prédit les réadmissions » sans « modestement » ; ne jamais le présenter comme outil clinique.
- Ne pas lire les coefficients de la régression logistique ou l'importance de la forêt comme des facteurs de risque ou des causes.
- Ne pas dire que les probabilités sont fiables : pas de calibration.
- Ne pas affirmer que le modèle est équitable : pas d'analyse par sous-groupes.
- Ne pas présenter les pistes (SMOTE, XGBoost, etc.) comme testées.
- Ne pas dire que les résultats sont généralisables (1999-2008, hôpitaux américains).

### (e) Phrase de conclusion (fin de l'exposé)

« Pour conclure : sur 99 343 séjours, nos modèles distinguent un peu mieux que le hasard les réadmissions à 30 jours (ROC-AUC ≈ 0,67), sans gagnant net entre HGB, forêt et régression logistique, avec environ 4 fausses alertes sur 5. C'est au mieux un outil de repérage, pas un outil clinique. Merci, nous sommes prêts pour vos questions. »

---

## 2. À savoir par TOUS

### 2.1 Glossaire

| Terme | Définition simple | Exemple chiffré du projet |
|---|---|---|
| Précision (precision) | Part des alertes qui sont de vrais cas | HGB : 0,181 ; environ 4 alertes sur 5 fausses à ≈ 0,2 |
| Rappel (recall) | Part des vrais cas détectés | HGB : 0,571 ; KNN base : 0,045 |
| F1 | Moyenne harmonique précision et rappel | HGB : 0,275 |
| ROC-AUC | Probabilité qu'un positif ait un score supérieur à un négatif ; 0,5 = hasard | HGB : 0,670 |
| PR-AUC | Aire sous la courbe précision-rappel, calculée comme la précision moyenne (`average_precision`) ; référence aléatoire = prévalence | HGB : 0,227 contre ≈ 0,11 au hasard |
| Seuil de décision | Probabilité au-dessus de laquelle on prédit 1 | 0,5 par défaut ; 0,535 optimisé pour HGB |
| `class_weight` | Pondère plus les erreurs sur la classe rare | `balanced` pour LR, arbre, forêt, HGB ; pas pour KNN |
| `GroupKFold` | Validation croisée où tous les séjours d'un patient restent dans un même pli | 5 plis groupés par `patient_nbr` |
| One-hot | Une colonne 0/1 par modalité | environ 2 400 colonnes après encodage |
| Imputation | Remplacer une valeur manquante par une valeur estimée | médiane (numérique), valeur la plus fréquente (catégorielle) |
| Standardisation | Centrer et réduire une variable (moyenne 0, écart-type 1) | `StandardScaler`, ajusté sur le train |
| PCA | Axes orthogonaux successifs qui capturent le plus de variance (sur les données standardisées) | PC1 25,5 % ; PC1+PC2 43,6 % |
| K-Means | Regroupe les points en k groupes autour de centres | k=3 ; clusters de 56 019, 14 140, 29 184 séjours |
| Silhouette | Mesure de séparation des clusters (de −1 à 1 ; ≈ 0 : clusters qui se chevauchent ; négatif : points mal affectés ; proche de 1 : bien séparés) | 0,187 pour k=3 : faible |
| Validation croisée | Entraîner/évaluer sur plusieurs découpages du train pour estimer la variabilité | HGB F1 0,279 ± 0,009 |
| Fuite de données (leakage) | Information indisponible au moment de la prédiction, ou issue des données d'évaluation, qui entre dans l'entraînement et gonfle le score | même patient dans train et test |
| Accuracy | Part de prédictions correctes | « toujours 0 » : 0,888 |

### 2.2 Méthodes vues au cours ou hors cours

| Méthode | Vue au cours ? | Rôle |
|---|---|---|
| `StandardScaler` | Oui | Mise à l'échelle |
| K-Means | Oui | Clustering exploratoire (k=3) |
| PCA | Oui | Projection, variance expliquée |
| Silhouette | Oui | Choix de k |
| Train / validation / test | Oui (notion) | Découpage |
| KNN | Oui, en régression seulement | Ici en classification (hors cours) |
| Validation croisée | Mentionnée | 5 plis |
| Régression logistique, arbre, forêt, HGB | Non | Modèles |
| Matrice de confusion, précision/rappel/F1, ROC-AUC, PR-AUC | Non | Métriques |
| `GroupKFold`, `RandomizedSearchCV`, `class_weight`, seuil, one-hot, imputation | Non | Protocole et préparation |

**Pourquoi des méthodes hors cours ?** La consigne du projet encourage explicitement les techniques hors cours. Nous les avons utilisées parce que le problème (classes déséquilibrées, patients répétés) l'exige (métriques adaptées, validation groupée), et nous les avons expliquées dans le rapport. Nous n'affirmons pas les maîtriser au-delà de ce qui est décrit ici.

### 2.3 Les 10 chiffres à connaître

| # | Chiffre | Source |
|---|---|---|
| 1 | 101 766 séjours, 71 518 patients | `eda_summary.txt` |
| 2 | 11,16 % de réadmissions < 30 j | `eda_summary.txt` |
| 3 | 2 423 exclus, 99 343 conservés | `README.md` / rapport |
| 4 | 79 541 train / 19 802 test | rapport |
| 5 | « Toujours 0 » : accuracy 0,888 | rapport (`y_test.csv`) |
| 6 | ROC-AUC ≈ 0,67 (HGB 0,670, forêt 0,662, LR 0,656 en base) | `model_comparison.csv` |
| 7 | CV HGB : F1 0,279 ± 0,009 ; ROC-AUC 0,669 ± 0,010 | `cv_results.csv` |
| 8 | Précision ≈ 0,2 pour un rappel ≈ 0,5 (≈ 4 alertes sur 5 fausses) | `tuned_comparison.csv` |
| 9 | `number_inpatient` : r = 0,165 | `eda_summary.txt` |
| 10 | Silhouette 0,187 (k=3), taux de 9,2 % à 17,1 % | `clustering_summary.txt` |

### 2.4 Les limites en 6 lignes

1. Scores modestes (ROC-AUC ≈ 0,67) : variables faiblement informatives.
2. Un seul split de test, pas d'intervalle de confiance ; petits écarts non significatifs.
3. Réglage limité (26 514 lignes, au plus 8 tirages, petites grilles) ; KNN sur 20 000 lignes.
4. Données de 1999-2008, hôpitaux américains ; imputation simple ; pas de calibration.
5. Pas d'analyse d'équité par sous-groupes ; certaines variables ne sont connues qu'à la sortie.
6. Pas un outil clinique : sans validation, gouvernance et revue d'experts, aucune décision réelle.

### 2.5 Lancer le pipeline

Depuis la racine du projet, dans cet ordre :

```text
python code/01_explore_data.py
python code/02_preprocessing.py
python code/03_train_models.py
python code/04_evaluation.py
python code/05_visualize_results.py
python code/06_pca_kmeans.py
python code/07_cv_tuning.py
```

- Avant, installer les dépendances : `python -m pip install -r requirements.txt`.
- Le fichier `diabetic_data.csv` doit être placé dans `data/` (non versionné).
- `07` dure environ 10 minutes (mesuré sur 22 cœurs) ; le temps des autres scripts n'est pas consigné.
- Relancer `04` et `05` après `07` pour intégrer les modèles réglés aux figures.
- `python code/07_cv_tuning.py --force` recalcule tous les résultats de 07 (à faire après une modification de 02, car l'empreinte ne couvre pas les tables de variables). `random_state=42`.
- Versions : Python 3.14, pandas 3.1.1, numpy 2.5.5, matplotlib 3.11.2, scikit-learn 1.9.1.

---

## 3. Répétition

**Minutage conseillé** (≈ 9 minutes hors questions)

| Partie | Durée | Slides |
|---|---|---|
| A | ≈ 3 min | 1-4 (cover 20 s, objectif 50 s, données 50 s, exploration 60 s) |
| B | ≈ 3 min | 5-9 (protocole 50 s, modèles 40 s, résultats 40 s, CV 30 s, courbes 20 s) |
| C | ≈ 3 min | 10-14 (réglage 50 s, variables 30 s, analyse 40 s, limites 20 s, suites 30 s) |
| Questions | libre | tout le monde peut répondre |

Ce minutage est une proposition, pas une contrainte de l'énoncé : adaptez-le à la durée imposée (non précisée dans les fichiers).

**Trois conseils**
1. Dites les chiffres arrondis comme sur la slide (0,67 ; 0,2 ; 11 %) et ne lisez pas les tableaux : un message par slide.
2. Répétez les transitions à voix haute ; chronométrez une fois au complet.
3. Pour une question piégeuse, répondez par « ce que les résultats soutiennent » (modeste, un seul test, pas de gagnant net), et dites « nous ne l'avons pas mesuré » quand c'est le cas (équité, calibration, causalité).

---

## Sources des chiffres

- `report/rapport.md` (rapport vérifié, source principale)
- `README.md`
- `results/eda_summary.txt`, `results/clustering_summary.txt`, `results/clustering_summary.csv`
- `results/model_comparison.csv`, `results/cv_results.csv`, `results/tuned_comparison.csv`
- `results/best_params.json`, `results/thresholds.json`
- Via le rapport : comptages dans `results/y_train.csv`, `results/y_test.csv` et `results/predictions/*_predictions.csv` (comptages recontrôlés)
- Slides : `slides/*.html` (cover, objectif, donnees, exploration, protocole, modeles, resultats, cv, courbes, reglage, variables, analyse, limites, suites)

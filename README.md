# Film Yorumları Üzerinde Duygu Analizi

Bu projede 50.000 film yorumundan oluşan veri seti üzerinde doğal dil işleme ve duygu sınıflandırma çalışması gerçekleştirilmiştir.

## Projenin Amacı

Film yorumlarını pozitif ve negatif olarak sınıflandırmak, yorumlarda en sık kullanılan kelimeleri incelemek ve farklı makine öğrenmesi modellerinin performansını karşılaştırmak.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TensorFlow
- Keras
- Gensim
- Matplotlib
- Seaborn
- WordCloud

## Uygulanan Adımlar

1. Veri setinin yüklenmesi
2. Eksik değer kontrolü
3. HTML etiketlerinin temizlenmesi
4. Küçük harfe dönüştürme
5. Noktalama işaretleri ve özel karakterlerin kaldırılması
6. Stopword temizliği
7. Kelime sıklığı ve WordCloud analizi
8. TF-IDF ile metinlerin sayısallaştırılması
9. Naive Bayes, Logistic Regression ve Random Forest modellerinin eğitilmesi
10. Bidirectional LSTM modelinin geliştirilmesi
11. Confusion Matrix, Classification Report ve ROC-AUC ile değerlendirme
12. Word2Vec, PCA, LDA ve t-SNE ile kelime ve konu analizi

## Model Sonuçları

Bidirectional LSTM modeli yaklaşık **%85,77 test doğruluğu** elde etmiştir. Eğitim sırasında doğrulama doğruluğu bazı epoch'larda yaklaşık **%88,36** seviyesine ulaşmıştır.

## Kurulum

```bash
pip install pandas numpy nltk scikit-learn tensorflow gensim matplotlib seaborn wordcloud
```

NLTK stopword paketini indirin:

```python
import nltk
nltk.download("stopwords")
```

## Çalıştırma

Veri setini `data` klasörüne ekleyin:

```python
df = pd.read_csv("data/Proje3.csv")
```

Ardından notebook dosyasını sırayla çalıştırın.

## Geliştirme Fikirleri

- EarlyStopping kullanmak
- Hiperparametre optimizasyonu yapmak
- Transformer tabanlı modeller denemek
- Streamlit arayüzü ile canlı duygu tahmini eklemek

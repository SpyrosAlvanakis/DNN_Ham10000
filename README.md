# **HAM10000 Skin Lesion Analysis Repository**

HAM10000 dataset encompasses 10,000 high-quality images of skin lesions, serving as a crucial resource for dermatological research and machine learning applications. Our objective is to extract maximum insight from this dataset through meticulous preprocessing, augmentation, and the implementation of cutting-edge machine learning models.

## **Key Features**

    Data Preprocessing and Augmentation: We have preprocessed the dataset to ensure optimal model performance. Our preprocessing steps include reshaping the images, adding Gaussian noise to enhance model robustness, and balancing the dataset to reduce the effects of data imbalance.

    Machine Learning Models and Strategies:
        Custom CNN Model: We designed a convolutional neural network (CNN) from scratch, focusing on hyperparameter tuning to optimize performance.
        Pretrained Models: Use pretrained models, we adapted them to our specific task, achieving significant accuracy improvements.
        Scikit-Learn with Dimensionality Reduction: We employed Scikit-Learn libraries, incorporating dimensionality reduction techniques and partial fitting to train simpler models efficiently.

    Frameworks Used: This project utilizes PyTorch and Scikit-Learn for model development and classification tasks, highlighting our flexible approach to apply different machine learning libraries.

    High Accuracy Achievements: Our custom CNN model achieved an impressive accuracy of nearly 95%, while the application of pretrained models pushed the accuracy to almost 99%.

    Resource Efficiency: A crucial aspect of our project was to maintain a high level of computational efficiency, with a goal to use less than 16GB of RAM throughout the process, making our methods accessible to researchers with limited hardware resources.


## Image-Sentiment Classification Using advertisement-sentiment dataset

#### dataset : http://people.cs.pitt.edu/~kovashka/ads/ (Automatic Understanding Of Image And Video Advertisements)
The dataset consist of a total of 30,430 advertisement images and 30 multiple sentiment labels.

![image](https://user-images.githubusercontent.com/60679596/163516312-6125c8ed-1e4c-4e08-b006-625d0676c35b.png)

</br>

#### data loader : https://gist.github.com/kyamagu/0aa8c06501bd8a5816640639d4d33a17

`dataloader.py`


</br>
#### model : DNN, ResNet50

`model.py`
</br>
</br>
### How to use

```python
python main.py --num_epochs(int) --device(str) --optimizer(str) --arch(str)
```


</br>
</br>

## Constructing perturbation image dataset
`Constructing_perturbation_images.py`

<img src="https://user-images.githubusercontent.com/60679596/163770119-a2a232dc-aef2-419e-8749-d8630ebb9dd8.png" width="450" height="750">


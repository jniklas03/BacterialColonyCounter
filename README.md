# PHASE: Phenotypic Analysis of Starvation Events
A Python package for the image-based quantification of distinct lag phases during colony growth of carbon-starved _E. coli_. This is a part of my Bachelors Thesis: **INSERT LINK**.

## Theory

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam tempus, nulla ut porttitor dapibus, erat orci volutpat quam, id ornare felis nisl dapibus ipsum. Curabitur id nisi vitae lectus interdum viverra. Quisque hendrerit velit quis eleifend molestie. Sed eu metus ac leo placerat efficitur non blandit erat. Nulla ornare orci et enim eleifend venenatis. Phasellus id tempus neque. Integer at cursus enim, eu lacinia purus. Vestibulum mauris ligula, congue et ante in, convallis tristique massa. Nulla suscipit fermentum efficitur. Aenean ullamcorper, purus a condimentum venenatis, nisl elit bibendum lacus, eget fermentum tellus ex eget magna. Etiam ut lectus condimentum, consectetur nisi ut, semper justo. Aliquam vel est ligula. Cras porttitor arcu mauris, in aliquet orci tincidunt quis. Nam quis orci vel mauris rhoncus vehicula sit amet sed tellus.

Nulla porttitor velit imperdiet nibh efficitur tincidunt. Pellentesque eget mi eros. Aenean ut vulputate magna. Vivamus at lacus rutrum, lobortis ipsum auctor, aliquam orci. Vestibulum in tristique diam. Pellentesque euismod pulvinar dictum. Fusce vel enim quis purus vulputate hendrerit ut quis tellus. Phasellus nec congue nisi. Maecenas efficitur augue ligula, sit amet placerat urna tincidunt at. In eu laoreet diam. Quisque eu dolor euismod, ornare nunc a, interdum mauris. Proin pretium lorem leo, a dictum mi porttitor nec. Fusce ornare efficitur faucibus. Integer velit risus, fermentum eu magna ut, tincidunt venenatis arcu. In porttitor malesuada ligula in eleifend.

Donec id sem id dolor consectetur ultricies vel ac turpis. Vestibulum ex turpis, tempus nec ultrices sit amet, tincidunt eu erat. Nulla efficitur magna tortor, sed vehicula orci rutrum ut. Praesent vel tincidunt lectus. Nullam euismod, nibh vel euismod congue, leo nisi dapibus elit, eget lacinia sapien enim id nulla. Quisque sodales rutrum ipsum, nec mollis risus bibendum ac. Mauris luctus gravida augue a pellentesque. Interdum et malesuada fames ac ante ipsum primis in faucibus. Proin a velit et ipsum congue lobortis nec et nunc. Aliquam enim erat, hendrerit vitae accumsan nec, fermentum ac eros. Nunc tincidunt suscipit aliquet. Praesent at dictum risus, et volutpat est. Integer a elit ligula. Fusce iaculis bibendum maximus. Proin ornare lacus molestie arcu blandit, a euismod magna lacinia. Duis eu luctus eros, eu rhoncus tortor.

Integer sit amet iaculis nisi, nec fermentum dui. Suspendisse sem dolor, commodo ut tellus vitae, scelerisque posuere nulla. Phasellus lobortis nec diam ut cursus. Donec dapibus massa eu ligula imperdiet posuere. Nullam efficitur magna at ipsum volutpat dapibus. Aenean vulputate odio eget ligula pellentesque pulvinar. Cras nisi ante, rutrum id vehicula nec, dapibus vel risus. Sed sit amet lorem dignissim, pellentesque neque eget, fringilla odio. Praesent laoreet venenatis nisi mollis tempor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam id auctor odio.

## Demonstration

### Cropping
The dish detection and cropping module is based on the OpenCV implementation of Hough Circle Transform. It takes an input image of `n` petri dishes and returns crops of each petri dish.<br />
![dish_detection](https://github.com/user-attachments/assets/f779750b-5d15-4f07-99ff-20e7d450e7b7)

### Preprocessing
The preprocessing is modular. Firstly, it contains two functions for mask creation for use in timelapse pipelines, `preprocess_fg_isolation` and `preprocess_bg_isolation`. They use first and last images as ground truth for the foreground (colonies) and background (artifacts) respectively. Finally the core preprocessing function, which:
- isolates the green channel of the input image
- thresholds it using OpenCV's adaptive thresholding
- filters for large artifacts using area selection using OpenCV's connected components
- erodes to remove further noise and artefacts and to separate touching colonies
- optionally applies the foreground and background masks<br />
![preprocessing](https://github.com/user-attachments/assets/1031dea3-fe61-4884-a9a4-17194d918977)

### Counting
The colony detection is done using OpenCV's `BlobDetector`.
<br />

![colony_detectino](https://github.com/user-attachments/assets/de310c92-85e1-4df9-9f02-a6fb837af227)

## Installation
Integer sit amet iaculis nisi, nec fermentum dui. Suspendisse sem dolor, commodo ut tellus vitae, scelerisque posuere nulla. Phasellus lobortis nec diam ut cursus. Donec dapibus massa eu ligula imperdiet posuere. Nullam efficitur magna at ipsum volutpat dapibus. Aenean vulputate odio eget ligula pellentesque pulvinar. Cras nisi ante, rutrum id vehicula nec, dapibus vel risus. Sed sit amet lorem dignissim, pellentesque neque eget, fringilla odio. Praesent laoreet venenatis nisi mollis tempor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam id auctor odio.


```python
pip install git+https://github.com/jniklas03/PHASE/
```

## Usage

Integer sit amet iaculis nisi, nec fermentum dui. Suspendisse sem dolor, commodo ut tellus vitae, scelerisque posuere nulla. Phasellus lobortis nec diam ut cursus. Donec dapibus massa eu ligula imperdiet posuere. Nullam efficitur magna at ipsum volutpat dapibus. Aenean vulputate odio eget ligula pellentesque pulvinar. Cras nisi ante, rutrum id vehicula nec, dapibus vel risus. Sed sit amet lorem dignissim, pellentesque neque eget, fringilla odio. Praesent laoreet venenatis nisi mollis tempor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam id auctor odio.

# matsciseg
materials science image segmentation algorithms

# Dependencies

* Python 2.7 (go [here][1] for instructions to setup python 2.7 on
  machines with older versions of python)
  
* OpenCV with Python bindings (tested with 2.3.1 and 2.4.3)

* All packages under `requirements.txt`, preferably installed into a
  virtualenv.  A bug in some numpy/scipy package dependencies might
  mandate that numpy be installed manually

* (optional) wxWidgets (must be manually compiled and installed into
  virtualenv) to use the now-deprecated graphical interface
  
* Tested on Debian Stable (Squeeze), Fedora 15, Arch Linux

# Setup

    virtualenv --no-site-packages ~/.virtualenv
    source ~/.virtualenv/bin/activate
    # compile/install opencv and make it available to the virutalenv
    pip install numpy==1.6.2 # fix circular dependency
    pip install -r requirements.txt
    make # fetches submodules, builds gco, and compiles gco wrappers
    
# Use

A command line interface (`matscicli.py`) and a web interface
(`webgui/manage.py runserver`) are available.

The CLI relies upon different "recipes" in the `recipes.py` file,
which can be easily customized as needed.  Its arguments are
documented upon invocation.  The "label" files are plain text files
with rows (separated by newlines) of integers (separated by
whitespace) for each row of pixels in the corresponding image.  Sample
invocations are shown in `tests/tests.sh`.

The web interface requires volumes made of sequences of images each
with a label initialization (same format as above) converted into
individual numpy `*.npz` files.  This can be done with the included
`wrap_labels.py` script.  Placing these in
`webgui/data/[data-name]/*.npz` will allow the web interface to find
the data.  To initialize the web interface, simply navigate to the
`webgui` folder, run

    ./manage runserver
    
and navigate to [http://localhost:8000](http://localhost:8000).

# References

Techniques used in this package are described in:

- Jarrell Waggoner, Youjie Zhou, Jeff Simmons, Ayman Salem, Marc De
  Graef, Song
  Wang. [Interactive Grain Image Segmentation using Graph Cut Algorithms][22],
  *Proceedings of SPIE Volume 8657 (Computational Imaging XI)*,
  Burlingame, CA, 2013

- Jarrell Waggoner, Jeff Simmons, Marc De Graef, and Song Wang.
  [Graph Cut Approaches for Materials Segmentation Preserving Shape, Appearance, and Topology][20].
  *International Conference on 3D Materials Science (3DMS)*, 147-152,
  Seven Springs, PA, 2012.

- Jarrell Waggoner, Jeff Simmons, and Song Wang.
  [Combining global labeling and local relabeling for metallic image segmentation][8].
  In *Proceedings of SPIE (Computational Imaging X)*, volume 8296,
  Burlingame, CA, 2012.
  
- Song Wang, Jarrell Waggoner, and Jeff Simmons.
  [Graph-cut methods for grain boundary segmentation][9].  *JOM
  Journal of the Minerals, Metals and Materials Society*, 63(7):49–51,
  2011.

---

Jarrell Waggoner  
/-/ [malloc47.com](http://www.malloc47.com)

[1]: http://www.malloc47.com/pythonbrew-opencv-debian/

# Documentation for AI Workshop

To build the HTML Page:
```
pandoc ANN-Basics.md -s --mathjax -t html5 -o html/ANN-Basics.html -M title=ANN\ Basics -c github.css
```

To build the PDF:
```
pandoc ANN-Basics.md -o pdf/ANN-Basics.pdf -V papersize:a4 -V geometry:margin=1in
```

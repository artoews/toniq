# 3D Printing

This folder includes three file formats (*.STL, *.html, and *.3mf) for each printed component of the phantom. Each format and its use case are described below.

*.STL files provide triangulated surface models of the components. These can serve as input files to a slicing application such as Cura.

*.html files provide a complete specification of the Cura settings used to print each component on a given printer, i.e. Ultimaker S5 (UMS5). These files were generated using the [Export HTML Cura Settings](https://marketplace.ultimaker.com/app/cura/plugins/5axes/HTMLCuraSettings) plug-in. Follow these steps for a quick view of just the user-modified print settings:
  1. Open the HTML file in a browser.
  2. Click "Visible settings".
  3. Click "User Modifications".

*.3mf files provide the complete Cura project used to print each component on a given printer, i.e. Ultimaker S5 (UMS5). This format is most useful for repeating a print job with (optional) incremental changes.

"""Builds a short synthetic technical report (4 pages) for local testing. Not a real technology."""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

PAGES = [
 ("1. Introduction and principle",
  "The soil-moisture sensor node is based on the principle of capacitive sensing: the dielectric "
  "permittivity of soil changes with its water content. This scientific principle is well established. "
  "The proposed application is automatic irrigation scheduling for small rice farms."),
 ("2. Laboratory results",
  "A breadboard of the sensing circuit was tested in the laboratory with five soil samples. "
  "Laboratory measurements showed a monotonic response between 5 and 45 percent volumetric water content. "
  "The proof-of-concept met its goal of detecting a 5 percent change."),
 ("3. Field-like testing",
  "A representative prototype was demonstrated in a relevant environment: an environmental chamber at "
  "35 C and 80 percent humidity. The test conditions and the configuration of the prototype are listed in Table 2. "
  "Measured accuracy was within the specified tolerance of 3 percent. The acceptance criteria were met."),
 ("4. Summary",
  "The management summary states that the system has reached TRL 6. "
  "No pilot installation in a farm has been carried out yet."),
]
c = canvas.Canvas("sample_report.pdf", pagesize=A4)
for i, (title, body) in enumerate(PAGES, 1):
    c.setFont("Helvetica-Bold", 14); c.drawString(56, 780, f"Soil-Moisture Sensor Node - Test Report (page {i})")
    c.setFont("Helvetica-Bold", 12); c.drawString(56, 750, title)
    c.setFont("Helvetica", 11)
    y = 725
    for line in simpleSplit(body, "Helvetica", 11, 480):
        c.drawString(56, y, line); y -= 15
    c.showPage()
c.save()

Scenario files
==============

Each CSV contains the prompts used in the audit, organized by:
  - dilemma (MF / YO / HL)
  - prompting regime (PLAIN / CC / ZSCOT / FSCOT)
  - language (EN / CN / JP / ES)

Filename convention: [DILEMMA]_[PROMPTING]_[LANGUAGE].csv

Total: 48 prompt files.

Columns: scenario_id, domain, title, scenario_text

Note on encoding
----------------
All files are UTF-8 encoded.

Non-English scenario files (CN/JP/ES) should be opened in a plain text
editor (e.g., VS Code, Sublime Text, Notepad++) or read directly via
pandas. Microsoft Excel may misinterpret the encoding and corrupt the
characters on display.

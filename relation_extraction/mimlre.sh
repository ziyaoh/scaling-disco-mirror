#!/bin/sh

cd ../MIML_RE/
ant all
sh run.sh edu.stanford.nlp.kbp.slotfilling.MultiR -props config/multir/multir_mimlre.properties
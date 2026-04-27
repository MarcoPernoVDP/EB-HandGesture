# EB-HandGesture - Neuromorphic Gesture Recognition (SNN)

## 1) Contesto del progetto
Questo progetto ha come obiettivo la realizzazione di un modello **Spiking Neural Network (SNN)** per il riconoscimento di gesti della mano a partire da dati a eventi, usando il dataset **EB-HandGesture**.

Il flusso e' pensato per una pipeline pienamente neuromorfica:
- dati a eventi (event-based vision),
- modello SNN sviluppato in **sinabs**,
- deployment su hardware neuromorfico **Speck** (SynSense).

L'idea centrale e' progettare una rete che non sia solo accurata in simulazione, ma anche realmente compatibile con i vincoli del dispositivo target.

## 2) Obiettivo tecnico
L'obiettivo della consegna e' costruire un modello SNN "ottimale" rispetto a metriche task-specific, attraverso:
- identificazione e implementazione di architetture SNN adatte a Speck,
- esplorazione sistematica degli iperparametri,
- confronto tra configurazioni in termini di accuratezza e costo computazionale/energetico.

In pratica, l'ottimo non e' solo "massima accuracy", ma un compromesso tra:
- performance di classificazione,
- latenza inferenziale,
- complessita' del modello,
- aderenza ai limiti hardware.

## 3) Materiale di riferimento
Riferimenti forniti per il progetto:
- Sinabs + hardware documentation (Speck):
  - https://sinabs.readthedocs.io/v3.1.3/
- Dataset EB-HandGesture:
  - https://drive.google.com/file/d/1s2MhiM-6P4IckIa3Sb9L0dBlzKdAIu6H/view
- Descrizione dataset e esempio di utilizzo:
  - https://ieeexplore.ieee.org/document/10650870

Nota: durante il design del modello vanno sempre considerate le limitazioni architetturali e operative dell'hardware Speck.

## 4) Stato attuale del repository
Al momento il repository contiene:
- dipendenze in `requirements.txt`,
- dataset gia' estratto in `data/` con split `train/`, `val/`, `test/`.

Nella cartella dataset sono presenti file `.h5` (event stream) e file `_bbox.npy` (bounding box/metadata associati), ad esempio:
- `D_far_wave1.h5`
- `D_far_wave1_bbox.npy`

Le classi osservabili dai nomi file includono (almeno):
- `armroll`, `clap`, `fist`, `point`, `wave`

Sono inoltre presenti varianti di scenario come `near` e `far`.

## 5) Background sintetico (per lettore esterno)
### 5.1 Event-based vision
A differenza delle immagini frame-based, i sensori a eventi registrano cambiamenti di luminanza in modo asincrono. Questo produce stream sparsi nel tempo, con vantaggi su:
- latenza,
- efficienza energetica,
- robustezza al motion blur.

### 5.2 Spiking Neural Networks
Le SNN processano informazione come spike discreti nel tempo. Sono naturalmente adatte ai dati a eventi e ai chip neuromorfici, ma richiedono progettazione temporale (finestre, dinamiche neuronali, soglie, leak, ecc.).

### 5.3 Perche' Speck + sinabs
- **sinabs** permette sviluppo e simulazione di reti spike-based in ambiente PyTorch-like.
- **Speck** e' il target hardware reale, quindi la rete deve essere "hardware-aware" sin dalla fase di progettazione.

## 6) Come si dovrebbe svolgere il progetto (pianificazione)
Di seguito una roadmap consigliata, con output attesi per ogni fase.

### Fase A - Data understanding e protocollo
1. Esplorare struttura reale del dataset:
   - numero campioni per split,
   - distribuzione classi,
   - durata media delle sequenze,
   - differenze near/far.
2. Definire protocollo di input:
   - binning temporale (numero di time-bin o window size),
   - risoluzione spaziale effettiva,
   - eventuale cropping/ROI via bbox.
3. Stabilire metriche ufficiali:
   - accuracy top-1,
   - macro-F1 (se class imbalance),
   - latenza media (ms),
   - stima costo computazionale/spike activity.

**Deliverable fase A:** report EDA + specifica preprocessing/versione del dataset usata.

### Fase B - Baseline SNN riproducibile
1. Implementare una baseline semplice ma robusta (es. CNN spiking con pochi layer).
2. Definire training loop completo:
   - loss + optimizer,
   - scheduler,
   - seed e riproducibilita',
   - checkpointing.
3. Valutare su validation set e registrare curve principali.

**Deliverable fase B:** baseline funzionante con metriche di riferimento.

### Fase C - Co-design con vincoli hardware Speck
1. Verificare compatibilita' layer/operazioni con Speck.
2. Inserire vincoli nella progettazione:
   - profondita'/larghezza rete,
   - shape dei feature map,
   - tipi di neuroni/soglie supportate,
   - budget di spike/attivazioni.
3. Validare che il modello sia deployable (non solo simulabile).

**Deliverable fase C:** shortlist di architetture realmente deployabili.

### Fase D - Hyperparameter optimization mirata
Impostare una ricerca strutturata (grid o random search guidata) sui parametri piu' influenti:
- learning rate,
- time bins / window size,
- threshold e leak neuronale,
- batch size,
- weight decay,
- ampiezza rete (canali/layer).

Per ogni trial salvare:
- metriche di accuratezza,
- costo (tempo training/inferenza),
- indicatori di attivita' spiking.

**Deliverable fase D:** tabella comparativa completa e ranking configurazioni.

### Fase E - Valutazione finale e analisi errori
1. Congelare la configurazione migliore su validation.
2. Valutare su test set una sola volta (o con protocollo controllato).
3. Analizzare confusion matrix e failure cases:
   - classi facilmente confondibili,
   - sensibilita' near/far,
   - impatto della durata sequenza.

**Deliverable fase E:** modello finale + analisi critica dei limiti.

## 7) Criterio di selezione del "modello ottimale"
Per evitare decisioni arbitrarie, usare una funzione di scoring multi-obiettivo, ad esempio:

$$
Score = \alpha \cdot Accuracy + \beta \cdot F1 - \gamma \cdot Latency - \delta \cdot Complexity
$$

con pesi $\alpha, \beta, \gamma, \delta$ scelti in base alle priorita' del progetto/corso.

## 8) Rischi tecnici principali e mitigazioni
- **Rischio:** modello accurato ma non mappabile su Speck.
  - **Mitigazione:** verifica compatibilita' hardware in ogni iterazione, non solo alla fine.
- **Rischio:** overfitting su pochi soggetti/sequenze.
  - **Mitigazione:** validazione rigorosa, seed multipli, augment temporali moderati.
- **Rischio:** risultati non riproducibili.
  - **Mitigazione:** logging completo config, seed fissi, versionamento esperimenti.

## 9) Ambiente software
Dipendenze principali (gia' in `requirements.txt`):
- PyTorch (`torch`, `torchvision`, `torchaudio`)
- `sinabs`
- `numpy`, `pandas`, `matplotlib`, `h5py`, `scipy`, `tqdm`

Installazione tipica:

```bash
pip install -r requirements.txt
```

## 10) Risultato atteso del progetto
Un output finale credibile dovrebbe includere:
- una pipeline riproducibile di preprocessing + training + evaluation,
- almeno una baseline e una/e configurazione/i ottimizzata/e,
- giustificazione chiara delle scelte architetturali,
- evidenza quantitativa del compromesso performance-vincoli hardware,
- proposta finale pronta per deployment neuromorfico (o con gap residui esplicitati).

---

Questo README definisce il **background** e la **pianificazione operativa** del progetto. Le fasi implementative (codice training, benchmark, deployment) verranno sviluppate nei prossimi step.

import os
import shutil
import concurrent.futures
from PyQSPICE import clsQSPICE as pqs
import matplotlib.pyplot as plt
import time
import glob
import re


class QSPICEBatchRunner:
    """
    Batch-Runner für QSPICE-Simulationen mit PyQSPICE.
    Führt Simulationen parallel aus und sammelt Ergebnisse sequentiell ein.
    """

    def __init__(self, basefile: str, workdir: str = "batch_runs"):
        """
        basefile: Name ohne Endung (Basis für .qsch/.cir)
        workdir: Hauptarbeitsverzeichnis für alle Batch-Simulationsläufe
        """
        self.basefile = basefile
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)

    def qsch_to_cir(self, qsch_path: str):
        """
        Erzeugt aus einer .qsch-Datei das zugehörige .cir-File.
        Kopiert das .qsch, falls nötig, ins Arbeitsverzeichnis.
        Gibt Pfad zur erzeugten .cir-Datei zurück.
        """
        base = os.path.splitext(os.path.basename(qsch_path))[0]
        target_qsch = os.path.join(self.workdir, os.path.basename(qsch_path))
        if not os.path.exists(target_qsch):
            shutil.copy2(qsch_path, target_qsch)
        cwd_before = os.getcwd()
        os.chdir(self.workdir)
        run = pqs(base)
        run.qsch2cir()
        cir_path = os.path.join(self.workdir, f"{base}.cir")
        os.chdir(cwd_before)
        if not os.path.isfile(cir_path):
            raise FileNotFoundError(f".cir Datei wurde nicht erzeugt: {cir_path}")
        return cir_path

    def change_component_value_in_lines(self, lines, component_name, new_value):
        """Change value in a list of lines (used for temp file creation)."""
        for i, line in enumerate(lines):
            if line.strip().startswith(component_name):
                parts = line.strip().split()
                if len(parts) >= 4:
                    parts[3] = str(new_value)
                    lines[i] = ' '.join(parts) + '\n'
                break
        return lines
    



    def change_parameter_value_in_lines(self, lines, parameter_name, new_value):
        """
        Robust ersetzt den Wert eines Parameters in .param-Zeilen.
        Unterstützt mehrere Parameter pro Zeile, Sonderzeichen wie µ und Leerzeichen um '='.

        Args:
            lines (list of str): Liste der Zeilen aus der SPICE-Datei.
            parameter_name (str): Der zu ersetzende Parameter.
            new_value (float): Der neue Wert für diesen Parameter.

        Returns:
            list of str: Die modifizierten Zeilen.
        """
        updated_lines = []
        formatted_value = f"{new_value:.15g}"

        for line in lines:
            new_line = line
            if line.strip().lower().startswith(".param"):
                try:
                    l = line.replace('.param ', '')
                    lis = l.split('=')
                    if lis[0].strip() == (parameter_name):
                        lis[1] = formatted_value
                        new_line = f".param {lis[0]} = {lis[1]} \n"                                     
                except Exception as e:
                    # Bei Fehler einfach Originalzeile übernehmen
                
                    print('Error in Paramteranpassung')
            updated_lines.append(new_line)
        return updated_lines



    def generate_param_cir_files(self, param_list, cir_template=None):
        """
        Erzeugt für jeden Parametersatz eine angepasste .cir Datei auf Basis eines Templates.
        param_list: Liste von Dictionaries, z.B. [{'R1': 1000, 'C1': 1e-6}, ...]
        cir_template: Optional, expliziter Pfad zu einem .cir-Template (sonst <basefile>.cir)
        Rückgabe: Liste der erzeugten .cir-Dateipfade
        """
        if cir_template is None:
            cir_template = os.path.join(self.workdir, f"{self.basefile}.cir")
        cir_files = []
        for idx, params in enumerate(param_list):
            cir_file = os.path.join(self.workdir, f"{self.basefile}_run{idx}.cir")
            with open(cir_template, "r") as f:
                content = f.readlines()
            for key, value in params.items():
                content = self.change_parameter_value_in_lines(content, key, value)
               # content = content.replace(f"${{{key}}}", str(value))
            with open(cir_file, "w") as f:
                f.writelines(content)
            cir_files.append(cir_file)
        return cir_files

    def _sim_only(self, cir_file, idx, max_attempts=3, retry_wait=0.5):
        """
        Startet nur die Simulation (cir2qraw), keine Daten werden geladen!
        - Stellt sicher, dass .cir vorhanden ist, bevor QSPICE läuft.
        - Nach dem Sim-Lauf wird geprüft, ob .qraw erzeugt wurde.
        - Wenn nicht, wird die Simulation bis zu max_attempts wiederholt.
        """
        try:
            base_name = os.path.splitext(os.path.basename(cir_file))[0]
            sim_dir = os.path.join(self.workdir, f"run_{idx}")
            os.makedirs(sim_dir, exist_ok=True)
            sim_cir_path = os.path.join(sim_dir, f"{base_name}.cir")
            shutil.copy2(cir_file, sim_cir_path)
            # Copy PWL voltage file if it exists
            voltage_file = os.path.join(
                self.workdir,
                "voltage.txt"
            )

            if os.path.exists(voltage_file):
                shutil.copy2(
                    voltage_file,
                    os.path.join(sim_dir, "voltage.txt")
                )

            # .dll-Dateien aus dem Quellordner kopieren
            source_dir = os.path.dirname(cir_file)
            dll_files = glob.glob(os.path.join(source_dir, "*.dll"))

            for dll_file in dll_files:
                shutil.copy2(dll_file, sim_dir)
            
            # Warten, bis .cir sicher vorhanden ist (max. 1s)
            for _ in range(10):
                if os.path.exists(sim_cir_path):
                    break
                time.sleep(0.1)
            else:
                raise FileNotFoundError(f"{sim_cir_path} wurde nicht kopiert!")

            qraw_file = os.path.join(sim_dir, f"{base_name}.qraw")
            cwd_before = os.getcwd()
            success = False
            last_error = None

            for attempt in range(1, max_attempts + 1):
                os.chdir(sim_dir)
                try:
                    run = pqs(base_name)
                    run.cir2qraw()
                    # Kleine Pause, damit das Filesystem Zeit zum Schreiben hat
                    time.sleep(0.3)
                except Exception as e:
                    last_error = e
                finally:
                    os.chdir(cwd_before)
                
                # Überprüfe ob qraw vorhanden ist (und sinnvoll groß ist)
                if os.path.isfile(qraw_file) and os.path.getsize(qraw_file) > 512:
                    success = True
                    break
                else:
                    print(f"[Warnung] QRAW {qraw_file} fehlt oder ist zu klein nach Versuch {attempt}. Wiederhole Simulation ...")
                    time.sleep(retry_wait)

            if not success:
                msg = f"QRAW {qraw_file} wurde nach {max_attempts} Versuchen nicht korrekt erzeugt."
                if last_error:
                    msg += f" Letzter Fehler: {last_error}"
                return {"cir": sim_cir_path, "status": "FAIL", "error": msg, "idx": idx}
            
            return {"cir": sim_cir_path, "status": "OK", "idx": idx}

        except Exception as e:
            return {"cir": cir_file, "status": "FAIL", "error": str(e), "idx": idx}

    def _collect_data(self, cir_file, signals, idx):
        """
        Lädt das Ergebnis aus der .qraw Datei im jeweiligen Run-Ordner.
        Gibt DataFrame (data) und ggf. sim-Metadaten zurück.
        """
        try:
            base_name = os.path.splitext(os.path.basename(cir_file))[0]
            sim_dir = os.path.join(self.workdir, f"run_{idx}")
            cwd_before = os.getcwd()
            os.chdir(sim_dir)
            run = pqs(base_name)
            df = run.LoadQRAW(signals)
            os.chdir(cwd_before)
            return {"cir": os.path.join(sim_dir, f"{base_name}.cir"), "status": "OK", "data": df, "sim": run.sim}
        except Exception as e:
            return {"cir": cir_file, "status": "FAIL", "error": str(e)}

    def run_batch(self, cir_files, signals=None, max_workers=10):
        """
        1. Führt alle Simulationen parallel aus (nur QSPICE-Aufruf, keine Daten sammeln).
        2. Sammelt anschließend sequentiell die Ergebnisse aus den QRAW-Dateien ein.
        signals: Liste der gewünschten Signale (z.B. ["I(R1)"])
        max_workers: Anzahl paralleler Simulationsprozesse
        Rückgabe: Liste von Ergebnis-Dictionaries
        """
        # 1. Parallel Simulationen starten
        print("Starting simulations in parallel..")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            sim_futures = [executor.submit(self._sim_only, cf, idx)
                           for idx, cf in enumerate(cir_files)]
            sim_results = []
            for future in concurrent.futures.as_completed(sim_futures):
                sim_results.append(future.result())
        print("All simulations completed.")

        # 2. Ergebnisse sequentiell einsammeln
        print("Collecting results...")
        data_results = []
        for idx, cf in enumerate(cir_files):
            result = self._collect_data(cf, signals, idx)
            data_results.append(result)
        print("All results collected.")
        return data_results

    def clean(self, filetypes=('cir', 'qraw', 'png')):
        """
        Löscht für das Batch alle erzeugten .cir, .qraw, .png Files etc.
        """
        for entry in os.listdir(self.workdir):
            full_path = os.path.join(self.workdir, entry)
            if os.path.isdir(full_path) and entry.startswith("run_"):
                shutil.rmtree(full_path, ignore_errors=True)
        for f in os.listdir(self.workdir):
            if any(f.endswith("." + ft) for ft in filetypes):
                try:
                    os.remove(os.path.join(self.workdir, f))
                except Exception:
                    pass

    def plot_sweep(self, results, signal_name, x_axis=None, xlabel=None, ylabel=None, title=None, legend_prefix="Sweep "):
        """
        Plottet das angegebene Signal aus allen Ergebnissen in ein gemeinsames Fenster.
        - signal_name: Name der Spalte (z.B. "I(R1)")
        - x_axis: Name der X-Achsen-Spalte (optional, default: erste Spalte im DataFrame)
        - xlabel, ylabel, title: Achsenbeschriftungen und Titel (optional)
        """
        plt.figure(figsize=(9, 6))
        for idx, res in enumerate(results):
            if res["status"] != "OK" or res["data"] is None:
                print(f"Warnung: Keine Daten für Run {idx} (Status: {res['status']})")
                continue
            df = res["data"]
            if x_axis is None:
                # Nehme erste Spalte als X-Achse
                x = df.iloc[:, 0]
                x_name = df.columns[0]
            else:
                if x_axis not in df.columns:
                    print(f"Warnung: X-Achse '{x_axis}' nicht gefunden für Run {idx}, nehme erste Spalte.")
                    x = df.iloc[:, 0]
                    x_name = df.columns[0]
                else:
                    x = df[x_axis]
                    x_name = x_axis
            if signal_name not in df.columns:
                print(f"Warnung: Signal '{signal_name}' nicht gefunden für Run {idx}, skip.")
                continue
            y = df[signal_name]
            label = f"{legend_prefix}{idx+1}"
            plt.plot(x, y, label=label)

        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlabel(xlabel if xlabel else x_name, fontsize=13)
        plt.ylabel(ylabel if ylabel else signal_name, fontsize=13)
        if title:
            plt.title(title)
        else:
            plt.title(f"QSPICE Sweep: {signal_name}")
        plt.tight_layout()
        plt.show()


    def get_result_df(self, results, idx):
        """
        Liefert den DataFrame einer bestimmten Simulation (per Index) aus der Ergebnisliste zurück.
        Gibt None zurück, falls Daten nicht vorhanden oder Status nicht OK.
        """
        if not (0 <= idx < len(results)):
            print(f"[Fehler] Index {idx} außerhalb der Ergebnisliste (0 bis {len(results)-1})")
            return None
        res = results[idx]
        if res.get("status") == "OK" and res.get("data") is not None:
            return res["data"]
        else:
            print(f"[Hinweis] Keine Daten für Simulation {idx}. Status: {res.get('status')}")
            return None



import xnat
import sys

server = "https://imaging-platform.diz-ag.med.ovgu.de"
user = "3efd0638-d27e-496d-b95a-320b0f93ef24"
password = "fkaGxFiKntJVbuUtC5FQL7TwwqW3zQBQn5sXmvq6q7RHCe7V0ptRj6eCS2Futx7M"

with xnat.connect(server=server, user=user, password=password) as session:
    project = None
    for pid, p in session.projects.items():
        name = p.name.lower()
        if ("lu117" in name or "lu177" in name) and "recon" in name:
            project = p
            break
            
    if project:
        print("Found project:", project.name)
        for sub in project.subjects.values():
            for exp in sub.experiments.values():
                for res in exp.resources.values():
                    if "dose" in res.label.lower() or "map" in res.label.lower():
                        print(f"Sub: {sub.label}, Exp: {exp.label}, Res: {res.label}")
                        for f in res.files.values():
                            print(f"  File: {f.name}")
                    else:
                        for f in res.files.values():
                            if "dose" in f.name.lower():
                                print(f"Sub: {sub.label}, Exp: {exp.label}, Res: {res.label}")
                                print(f"  File MATCH: {f.name}")

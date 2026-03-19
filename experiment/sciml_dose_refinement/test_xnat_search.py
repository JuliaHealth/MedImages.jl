import xnat

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
        print("Searching project:", project.name)
        found = False
        for sub in project.subjects.values():
            # Check subject resources
            for res in sub.resources.values():
                for f in res.files.values():
                    # print("Sub Resource File:", f.name)
                    if "dose" in f.name.lower() or "map" in f.name.lower():
                        print(f"FOUND MATCH in Sub {sub.label} res {res.label}: {f.name}")
                        found = True
            
            # Check experiments
            for exp in sub.experiments.values():
                if "dose" in exp.label.lower():
                    print(f"FOUND EXP MATCH {sub.label} exp {exp.label}")
                    found = True
                for res in exp.resources.values():
                    if "dose" in res.label.lower():
                        print(f"FOUND RES MATCH {sub.label} res {res.label}")
                        found = True
                    for f in res.files.values():
                        if "dose" in f.name.lower():
                            print(f"FOUND FILE MATCH {sub.label} file {f.name}")
                            found = True
        
        if not found:
            print("No matches found anywhere in the project.")

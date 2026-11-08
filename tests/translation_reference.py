"""20 reference sentences with hand-written German and French translations (test T4).

The sentences are everyday meeting speech, the kind this demo interprets. The references were
written for this project; they are one acceptable translation each, not the only one, so the
chrF score is a regression guard (it must not fall below the first-run baseline), not a
claim of absolute quality.
"""

from __future__ import annotations

from collections import Counter

REFERENCES: list[tuple[str, str, str]] = [
    (
        "Good morning, everyone, and welcome to the meeting.",
        "Guten Morgen zusammen und willkommen zum Meeting.",
        "Bonjour à tous et bienvenue à la réunion.",
    ),
    (
        "Could you please send me the report by Friday?",
        "Könnten Sie mir bitte den Bericht bis Freitag schicken?",
        "Pourriez-vous m'envoyer le rapport d'ici vendredi ?",
    ),
    (
        "The new version fixes several bugs and improves performance.",
        "Die neue Version behebt mehrere Fehler und verbessert die Leistung.",
        "La nouvelle version corrige plusieurs bogues et améliore les performances.",
    ),
    (
        "I think we should move the deadline by two weeks.",
        "Ich denke, wir sollten die Frist um zwei Wochen verschieben.",
        "Je pense que nous devrions repousser l'échéance de deux semaines.",
    ),
    (
        "Our customers have asked for a simpler way to share files.",
        "Unsere Kunden haben nach einer einfacheren Möglichkeit gefragt, Dateien zu teilen.",
        "Nos clients ont demandé un moyen plus simple de partager des fichiers.",
    ),
    (
        "The meeting starts at nine o'clock tomorrow morning.",
        "Die Besprechung beginnt morgen früh um neun Uhr.",
        "La réunion commence demain matin à neuf heures.",
    ),
    (
        "We are very happy with the results of the first quarter.",
        "Wir sind mit den Ergebnissen des ersten Quartals sehr zufrieden.",
        "Nous sommes très satisfaits des résultats du premier trimestre.",
    ),
    (
        "Please turn off your microphones when you are not speaking.",
        "Bitte schalten Sie Ihre Mikrofone aus, wenn Sie nicht sprechen.",
        "Veuillez couper votre micro lorsque vous ne parlez pas.",
    ),
    (
        "This system works without an internet connection.",
        "Dieses System funktioniert ohne Internetverbindung.",
        "Ce système fonctionne sans connexion Internet.",
    ),
    (
        "The translation appears on the screen a few seconds after I speak.",
        "Die Übersetzung erscheint wenige Sekunden nach meinen Worten auf dem Bildschirm.",
        "La traduction apparaît à l'écran quelques secondes après que j'ai parlé.",
    ),
    (
        "Do you have any questions before we continue?",
        "Haben Sie Fragen, bevor wir fortfahren?",
        "Avez-vous des questions avant que nous continuions ?",
    ),
    (
        "The budget for next year has not been approved yet.",
        "Das Budget für das nächste Jahr wurde noch nicht genehmigt.",
        "Le budget de l'année prochaine n'a pas encore été approuvé.",
    ),
    (
        "I will call you back as soon as I have more information.",
        "Ich rufe Sie zurück, sobald ich mehr Informationen habe.",
        "Je vous rappellerai dès que j'aurai plus d'informations.",
    ),
    (
        "Thank you for your patience while we fix the problem.",
        "Vielen Dank für Ihre Geduld, während wir das Problem beheben.",
        "Merci de votre patience pendant que nous résolvons le problème.",
    ),
    (
        "Every employee will receive a short training next month.",
        "Jeder Mitarbeiter erhält nächsten Monat eine kurze Schulung.",
        "Chaque employé recevra une courte formation le mois prochain.",
    ),
    (
        "The software is installed on each computer in the office.",
        "Die Software ist auf jedem Computer im Büro installiert.",
        "Le logiciel est installé sur chaque ordinateur du bureau.",
    ),
    (
        "We need to decide which supplier we want to work with.",
        "Wir müssen entscheiden, mit welchem Lieferanten wir zusammenarbeiten wollen.",
        "Nous devons décider avec quel fournisseur nous voulons travailler.",
    ),
    (
        "Can everyone in the back of the room hear me?",
        "Können mich alle hinten im Raum hören?",
        "Tout le monde au fond de la salle m'entend-il ?",
    ),
    (
        "The project was finished on time and under budget.",
        "Das Projekt wurde termingerecht und unter Budget abgeschlossen.",
        "Le projet a été terminé à temps et sous le budget prévu.",
    ),
    (
        "Let us take a short break and continue in ten minutes.",
        "Lassen Sie uns eine kurze Pause machen und in zehn Minuten weitermachen.",
        "Faisons une courte pause et reprenons dans dix minutes.",
    ),
]


def _ngrams(text: str, n: int) -> Counter[str]:
    letters = text.replace(" ", "")
    return Counter(letters[i : i + n] for i in range(len(letters) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, beta=2) in percent, averaged over n = 1..max_n."""
    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        hyp, ref = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not hyp or not ref:
            continue
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / sum(hyp.values()))
        recalls.append(overlap / sum(ref.values()))
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    return 100.0 * (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

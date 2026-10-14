// Top nav shared by index.html and budget.html. NAV is the one list of menu
// items; each page sets <body data-page="..."> and links to its own sections
// with a bare #hash so the current-section highlight can find them.
(function () {
  const PAGES = { index: "index.html", budget: "budget.html" };
  const NAV = [
    { page: "index", id: "section-rightsizing", label: "Scenarios" },
    { group: "Buildings", page: "index", items: [
      ["section-map", "Utilization map"], ["section-utilization", "Utilization"],
      ["section-seismic", "Seismic"], ["section-transport", "Transportation"]] },
    { group: "Enrollment", page: "index", items: [
      ["section-trends", "Trends"], ["section-sustainability", "7-year change"],
      ["section-enrollment-forecast", "2035 forecast"]] },
    { group: "Programs", page: "index", items: [
      ["section-ratios", "Class sizes"], ["section-dli", "DLI"], ["section-sped", "Special ed"]] },
    { group: "Budget", page: "budget", items: [
      ["section-flow", "Money in and out"], ["section-gap", "The yearly gap"],
      ["section-contracts", "Contracts"], ["section-schools", "Cost per student"],
      ["section-savings", "Closure savings"], ["section-methodology", "Budget methodology"]] },
    { group: "Data", page: "index", items: [
      ["section-scatters", "Scatters"], ["section-table", "Table"],
      ["section-charters", "Charters"], ["section-methodology", "Methodology"]] },
  ];

  const here = document.body.dataset.page || "index";
  const nav = document.getElementById("primary-nav");
  const toggle = document.getElementById("nav-toggle");
  const current = document.getElementById("nav-current");
  const href = (page, id) => (page === here ? "" : PAGES[page]) + "#" + id;
  nav.innerHTML = NAV.map(n => n.group
    // On a page other than the dashboard, its own group stays marked so readers see where they are.
    ? `<div class="nav-group"><button type="button" class="nav-group-btn${n.page === here && here !== "index" ? " here" : ""}" aria-expanded="false">${n.group}</button>` +
      `<div class="nav-menu">${n.items.map(([id, label]) => `<a href="${href(n.page, id)}">${label}</a>`).join("")}</div></div>`
    : `<a href="${href(n.page, n.id)}">${n.label}</a>`).join("");

  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  const groups = Array.from(nav.querySelectorAll(".nav-group"));
  const closeGroups = (except) => groups.forEach(g => {
    if (g === except) return;
    g.classList.remove("open");
    g.querySelector(".nav-group-btn").setAttribute("aria-expanded", "false");
  });
  nav.addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-group-btn");
    if (btn) {
      const g = btn.parentElement;
      closeGroups(g);
      btn.setAttribute("aria-expanded", g.classList.toggle("open") ? "true" : "false");
      return;
    }
    if (e.target.tagName === "A") {
      nav.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
      closeGroups();
      e.target.blur();
    }
  });
  document.addEventListener("click", (e) => { if (!nav.contains(e.target)) closeGroups(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeGroups(); });

  // Highlight the section in the middle of the viewport, and name it beside the brand.
  // Only same-page links (bare #hash) take part.
  const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
  const sections = links.map(a => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  const clearActive = () => {
    links.forEach(a => a.classList.remove("active"));
    nav.querySelectorAll(".nav-group-btn").forEach(b => b.classList.remove("active"));
  };
  const onScroll = () => {
    // Above the first section (the intro) there is no current section to name.
    const atTop = !sections.length || sections[0].getBoundingClientRect().top > innerHeight * 0.5;
    current.classList.toggle("show", !atTop && current.textContent !== "");
    if (atTop) clearActive();
  };
  if ("IntersectionObserver" in window && sections.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const link = links.find(a => a.getAttribute("href") === "#" + entry.target.id);
        if (!link) return;
        clearActive();
        link.classList.add("active");
        const btn = link.closest(".nav-group")?.querySelector(".nav-group-btn");
        btn?.classList.add("active");
        current.innerHTML = (btn ? btn.textContent + " › " : "") + "<b>" + link.textContent + "</b>";
      });
      onScroll();
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
    sections.forEach(s => io.observe(s));
  }
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();

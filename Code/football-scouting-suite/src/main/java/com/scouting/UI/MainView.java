package com.scouting.ui;

import com.scouting.data.model.Player;
import com.scouting.service.PlayerFilterRequest;
import com.scouting.service.StatFilterCriteria;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.combobox.MultiSelectComboBox;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.Icon;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.Scroller;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.value.ValueChangeMode;
import com.vaadin.flow.router.Route;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Questa classe rappresenta il palcoscenico principale dell'applicazione (la "View" in MVC).
 * Costruita con Vaadin, permette di scrivere interfacce web usando puro codice Java.
 * Qui assembliamo visivamente i componenti: la griglia dei giocatori, i campi di ricerca ed i bottoni.
 *
 * Un aspetto interessante è l'uso della mappa `columnConfigurations`: invece di un metodo gigantesco per configurare
 * le colonne, usiamo delle "Lambda Expressions" mappate per nome. Questo ci permette di accendere e spegnere gruppi
 * di colonne (es. "Show Passing Stats") dinamicamente, rendendo l'interfaccia reattiva alle preferenze dell'utente.
 */

@Route("")
public class MainView extends VerticalLayout {
    
    // Costanti per SonarQube
    private static final String BASIC_INFO = "Basic Info";
    private static final String PLAYING_TIME = "Playing Time";
    private static final String STANDARD_ATTACKING = "Standard Attacking";
    private static final String SHOOTING = "Shooting";
    private static final String EXPECTED_METRICS = "Expected Metrics (xG)";
    private static final String PASSING = "Passing";
    private static final String PROGRESSIVE_PLAY = "Progressive Play";
    private static final String POSSESSION = "Possession & Dribbling";
    private static final String DEFENSIVE = "Defensive Actions";
    private static final String GOAL_SHOT_CREATION = "Goal & Shot Creation";
    private static final String PER90_ATT = "Per 90 (Attacking)";
    private static final String PER90_GEN = "Per 90 (General Play)";
    private static final String PER90_DEF = "Per 90 (Defensive)";

    private static final String WIDTH_80 = "80px";
    private static final String WIDTH_90 = "90px";
    private static final String WIDTH_100 = "100px";
    private static final String WIDTH_110 = "110px";
    private static final String WIDTH_120 = "120px";
    private static final String WIDTH_130 = "130px";
    private static final String WIDTH_150 = "150px";

    private final transient ScoutingController controller;
    private final transient List<Player> allPlayers;
    
    private final transient Map<String, Runnable> columnConfigurations;
    private final transient Map<String, String> statFieldMapping;

    private final Grid<Player> grid;
    private final Span recordCount;
    
    private IntegerField minAgeField;
    private IntegerField maxAgeField;
    private TextField playerSearch;
    private TextField squadSearch;
    private ComboBox<String> compFilter;
    private ComboBox<String> nationFilter;
    private ComboBox<String> positionFilter;
    
    private MultiSelectComboBox<String> columnSelector;
    private VerticalLayout filterRowsLayout; 
    private List<FilterRowComponent> activeFilterRows; 

    public MainView(ScoutingController controller) {
        this.controller = controller;
        this.allPlayers = controller.getAllPlayers();
        
        this.columnConfigurations = new LinkedHashMap<>();
        this.statFieldMapping = new LinkedHashMap<>();
        
        setSizeFull(); 
        setPadding(true);
        setSpacing(false);
        
        H1 title = new H1("Football Player Scouting Suite (2025-26)");
        title.getStyle().set("font-size", "1.5rem");
        title.getStyle().set("margin-top", "0");
        title.getStyle().set("margin-bottom", "10px");
        add(title);
        
        grid = new Grid<>(Player.class, false);
        grid.setSizeFull(); 
        
        this.activeFilterRows = new ArrayList<>();
        
        populateStatMapping();
        populateColumnConfigurations();
        
        createToolbar();
        createFilters(); 
        configureDefaultColumns();
        
        recordCount = new Span();
        recordCount.getStyle().set("font-weight", "bold");
        recordCount.getStyle().set("font-size", "0.9rem");
        updateRecordCount();
        
        add(columnSelector, recordCount, grid);
        expand(grid); 
        
        grid.setItems(allPlayers);
    }
    
    private void populateStatMapping() {
        statFieldMapping.put("Goals", "goals");
        statFieldMapping.put("Assists", "assists");
        statFieldMapping.put("G+A", "gPlusA");
        statFieldMapping.put("xG (Expected Goals)", "xg");
        statFieldMapping.put("xAG (Expected Assists)", "xag");
        statFieldMapping.put("Shots", "shots");
        statFieldMapping.put("SoT (Shots on Target)", "sot");
        statFieldMapping.put("Key Passes", "keyPasses");
        statFieldMapping.put("Progressive Passes", "progPasses");
        statFieldMapping.put("Progressive Carries", "progCarries");
        statFieldMapping.put("Tackles", "tackles");
        statFieldMapping.put("Interceptions", "interceptions");
        statFieldMapping.put("Goals per 90", "goalsP90");
        statFieldMapping.put("Assists per 90", "assistsP90");
        statFieldMapping.put("xG per 90", "xgP90");
        statFieldMapping.put("Tackles per 90", "tacklesP90");
    }

    private void createToolbar() {
        columnSelector = new MultiSelectComboBox<>("Seleziona Metriche");
        columnSelector.setItems(columnConfigurations.keySet());
        columnSelector.select(BASIC_INFO); 
        columnSelector.setWidth("100%");
        columnSelector.setMaxWidth("600px");
        columnSelector.setPlaceholder("Scegli categorie statistiche...");
        columnSelector.addValueChangeListener(e -> updateColumnVisibility());
    }
    
    private void createFilters() {
        minAgeField = new IntegerField("Min Age");
        minAgeField.setValue(16); minAgeField.setWidth(WIDTH_80);
        
        maxAgeField = new IntegerField("Max Age");
        maxAgeField.setValue(50); maxAgeField.setWidth(WIDTH_80);
        
        playerSearch = new TextField("Player Name");
        playerSearch.setPlaceholder("Search...");
        playerSearch.setWidth("160px"); 
        playerSearch.setValueChangeMode(ValueChangeMode.LAZY);
        
        squadSearch = new TextField("Squad");
        squadSearch.setPlaceholder("Search squad...");
        squadSearch.setWidth("160px");
        squadSearch.setValueChangeMode(ValueChangeMode.LAZY);
        
        Set<String> competitions = allPlayers.stream().map(Player::getCompetition).filter(Objects::nonNull).collect(Collectors.toSet());
        compFilter = new ComboBox<>("Competition");
        compFilter.setItems(competitions);
        compFilter.setWidth("180px");
        compFilter.setClearButtonVisible(true);
        
        Set<String> nations = allPlayers.stream().map(Player::getNation).filter(Objects::nonNull).collect(Collectors.toSet());
        nationFilter = new ComboBox<>("Nation");
        nationFilter.setItems(nations);
        nationFilter.setWidth("140px");
        nationFilter.setClearButtonVisible(true);
        
        Set<String> positions = allPlayers.stream().map(Player::getPosition).filter(Objects::nonNull).collect(Collectors.toSet());
        positionFilter = new ComboBox<>("Position");
        positionFilter.setItems(positions);
        positionFilter.setWidth(WIDTH_110);
        positionFilter.setClearButtonVisible(true);

        HorizontalLayout headerFilters = new HorizontalLayout();
        headerFilters.setWidthFull();
        headerFilters.setDefaultVerticalComponentAlignment(Alignment.BASELINE);

        headerFilters.getStyle().set("flex-wrap", "wrap"); 
        headerFilters.getStyle().set("gap", "10px");
        
        headerFilters.add(minAgeField, maxAgeField, playerSearch, squadSearch, compFilter, nationFilter, positionFilter);

        filterRowsLayout = new VerticalLayout();
        filterRowsLayout.setPadding(false);
        filterRowsLayout.setSpacing(false);
        filterRowsLayout.setWidthFull();
        
        Scroller scroller = new Scroller(filterRowsLayout);
        scroller.setScrollDirection(Scroller.ScrollDirection.VERTICAL);
        scroller.setWidthFull();
        scroller.getStyle().set("max-height", WIDTH_150); 
        scroller.getStyle().set("border-bottom", "1px solid var(--lumo-contrast-10pct)");

        Button addFilterBtn = new Button("Add Statistic Filter", new Icon(VaadinIcon.PLUS));
        addFilterBtn.addClickListener(e -> addFilterRow());
        
        Button resetBtn = new Button("Reset All", new Icon(VaadinIcon.REFRESH));
        resetBtn.addThemeVariants(ButtonVariant.LUMO_ERROR, ButtonVariant.LUMO_TERTIARY);
        resetBtn.addClickListener(e -> resetFilters());
        
        HorizontalLayout actionsBar = new HorizontalLayout(addFilterBtn, resetBtn);
        actionsBar.setWidthFull();
        actionsBar.setJustifyContentMode(JustifyContentMode.BETWEEN);
        actionsBar.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        actionsBar.setPadding(false);
        actionsBar.getStyle().set("margin-top", "10px");

        minAgeField.addValueChangeListener(e -> updateList());
        maxAgeField.addValueChangeListener(e -> updateList());
        playerSearch.addValueChangeListener(e -> updateList());
        squadSearch.addValueChangeListener(e -> updateList());
        compFilter.addValueChangeListener(e -> updateList());
        nationFilter.addValueChangeListener(e -> updateList());
        positionFilter.addValueChangeListener(e -> updateList());
        
        VerticalLayout filtersContainer = new VerticalLayout();
        filtersContainer.setPadding(false);
        filtersContainer.setSpacing(false);
        
        filtersContainer.add(headerFilters, actionsBar, scroller);
        
        add(filtersContainer);
    }
    
    private void addFilterRow() {
        FilterRowComponent row = new FilterRowComponent(
            statFieldMapping, 
            this::updateList, 
            componentToRemove -> { 
                filterRowsLayout.remove(componentToRemove);
                activeFilterRows.remove(componentToRemove);
                updateList();
            }
        );
        activeFilterRows.add(row);
        filterRowsLayout.add(row);
    }
    
    private void updateList() {
        Integer minAge = minAgeField.getValue();
        Integer maxAge = maxAgeField.getValue();
        String playerName = playerSearch.getValue();
        String squad = squadSearch.getValue();
        String comp = compFilter.getValue();
        String nation = nationFilter.getValue();
        String position = positionFilter.getValue();

        List<StatFilterCriteria> statCriteriaList = new ArrayList<>();
        
        for (FilterRowComponent row : activeFilterRows) {
            StatFilterCriteria criteria = row.getCriteria();
            if (criteria != null) {
                statCriteriaList.add(criteria);
            }
        }

        PlayerFilterRequest request = new PlayerFilterRequest(
            minAge, maxAge, playerName, squad, comp, nation, position, statCriteriaList
        );

        List<Player> filtered = controller.searchPlayers(request);

        grid.setItems(filtered);
        recordCount.setText(String.format("Showing %d of %d players", filtered.size(), allPlayers.size()));
    }
    
    private void resetFilters() {
        minAgeField.setValue(16);
        maxAgeField.setValue(50);
        playerSearch.clear();
        squadSearch.clear();
        compFilter.clear();
        nationFilter.clear();
        positionFilter.clear();
        
        filterRowsLayout.removeAll();
        activeFilterRows.clear();
        
        columnSelector.select(BASIC_INFO);
        
        updateList();
    }
    
    private void updateRecordCount() {
        recordCount.setText("Visualizzando %d giocatori (Database 2025-26)".formatted(allPlayers.size()));
    }
    
    private String fmt(Double value) {
        return value != null ? "%.2f".formatted(value) : "-";
    }
    
    private void populateColumnConfigurations() {

        columnConfigurations.put(BASIC_INFO, () -> {
            grid.addColumn(Player::getName).setHeader("Player").setFrozen(true).setWidth("180px").setSortable(true).setResizable(true);
            grid.addColumn(Player::getAge).setHeader("Age").setWidth("70px").setSortable(true);
            grid.addColumn(Player::getPosition).setHeader("Position").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getSquad).setHeader("Squad").setWidth(WIDTH_150).setSortable(true).setResizable(true);
            grid.addColumn(Player::getNation).setHeader("Nation").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getCompetition).setHeader("Competition").setWidth(WIDTH_150).setSortable(true).setResizable(true);
        });

        columnConfigurations.put(PLAYING_TIME, () -> {
            grid.addColumn(Player::getMatches).setHeader("Matches").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getStarts).setHeader("Starts").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getMinutes).setHeader("Minutes").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getNinetyS())).setHeader("90s").setWidth(WIDTH_80).setSortable(true);
        });

        columnConfigurations.put(STANDARD_ATTACKING, () -> {
            grid.addColumn(Player::getGoals).setHeader("Goals").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getAssists).setHeader("Assists").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getgPlusA).setHeader("G+A").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getNpg).setHeader("Non-Pen Goals").setWidth(WIDTH_130).setSortable(true);
            grid.addColumn(Player::getPkMade).setHeader("PK Made").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getPkAttempted).setHeader("PK Att").setWidth(WIDTH_100).setSortable(true);
        });

        columnConfigurations.put(SHOOTING, () -> {
            grid.addColumn(Player::getShots).setHeader("Shots").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getSot).setHeader("SoT").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getSotPercentage())).setHeader("SoT%").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getgPerShots())).setHeader("G/Sh").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getgPerSot())).setHeader("G/SoT").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getAvgShotDist())).setHeader("Avg Dist").setWidth(WIDTH_110).setSortable(true);
        });

        columnConfigurations.put(EXPECTED_METRICS, () -> {
            grid.addColumn(p -> fmt(p.getXg())).setHeader("xG").setWidth(WIDTH_80).setSortable(true);
            grid.addColumn(p -> fmt(p.getNpxg())).setHeader("npxG").setWidth(WIDTH_80).setSortable(true);
            grid.addColumn(p -> fmt(p.getXag())).setHeader("xAG").setWidth(WIDTH_80).setSortable(true);
            grid.addColumn(p -> fmt(p.getNpxgPlusXag())).setHeader("npxG+xAG").setWidth(WIDTH_110).setSortable(true);
            grid.addColumn(p -> fmt(p.getgMinusXg())).setHeader("G-xG").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getNpgMinusXg())).setHeader("npG-xG").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getaMinusXag())).setHeader("A-xAG").setWidth(WIDTH_90).setSortable(true);
        });
        
        columnConfigurations.put(PASSING, () -> {
            grid.addColumn(Player::getPassesCompleted).setHeader("Pass Cmp").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getPassesAttempted).setHeader("Pass Att").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getPassesPercentage())).setHeader("Pass %").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getKeyPasses).setHeader("Key Passes").setWidth(WIDTH_110).setSortable(true);
            grid.addColumn(Player::getPassesFinalThird).setHeader("Pass Final 3rd").setWidth(WIDTH_130).setSortable(true);
            grid.addColumn(Player::getPassesPenArea).setHeader("Pass Pen Area").setWidth(WIDTH_130).setSortable(true);
            grid.addColumn(Player::getCrossesPenArea).setHeader("Cross Pen Area").setWidth(WIDTH_130).setSortable(true);
        });
        
        columnConfigurations.put(PROGRESSIVE_PLAY, () -> {
            grid.addColumn(Player::getProgCarries).setHeader("Prog Carries").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(Player::getProgPasses).setHeader("Prog Passes").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(Player::getProgPassesReceived).setHeader("Prog Rec").setWidth(WIDTH_120).setSortable(true);
        });
        
        columnConfigurations.put(POSSESSION, () -> {
            grid.addColumn(Player::getTouches).setHeader("Touches").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getTakeOnAttempted).setHeader("Dribble Att").setWidth(WIDTH_110).setSortable(true);
            grid.addColumn(Player::getTakeOnSucc).setHeader("Dribble Succ").setWidth(WIDTH_110).setSortable(true);
            grid.addColumn(p -> fmt(p.getTakeOnPercentage())).setHeader("Dribble %").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getCarries).setHeader("Carries").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(Player::getDispossessed).setHeader("Dispossessed").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(Player::getMiscontrols).setHeader("Miscontrols").setWidth(WIDTH_110).setSortable(true);
        });

        columnConfigurations.put(DEFENSIVE, () -> {
            grid.addColumn(Player::getTackles).setHeader("Tackles").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getTacklesWon).setHeader("Tkl Won").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getTacklesPercentage())).setHeader("Tkl %").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getInterceptions).setHeader("Int").setWidth(WIDTH_80).setSortable(true);
            grid.addColumn(Player::getTklPlusInt).setHeader("Tkl+Int").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getBlocks).setHeader("Blocks").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getClearances).setHeader("Clear").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(Player::getErrorsLeadingShot).setHeader("Errors").setWidth(WIDTH_90).setSortable(true);
        });

        columnConfigurations.put(GOAL_SHOT_CREATION, () -> {
            grid.addColumn(Player::getSca).setHeader("SCA").setWidth(WIDTH_80).setSortable(true);
            grid.addColumn(Player::getGca).setHeader("GCA").setWidth(WIDTH_80).setSortable(true);
        });

        columnConfigurations.put(PER90_ATT, () -> {
            grid.addColumn(p -> fmt(p.getGoalsP90())).setHeader("Goals/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getAssistsP90())).setHeader("Assists/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getgPlusAP90())).setHeader("G+A/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getNpgP90())).setHeader("npg/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getXgP90())).setHeader("xG/90").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getXagP90())).setHeader("xAG/90").setWidth(WIDTH_90).setSortable(true);
        });

        columnConfigurations.put(PER90_GEN, () -> {
            grid.addColumn(p -> fmt(p.getShotsP90())).setHeader("Shots/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getSotP90())).setHeader("SoT/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getKeyPassesP90())).setHeader("Key Pass/90").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(p -> fmt(p.getProgCarriesP90())).setHeader("Prog Carr/90").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(p -> fmt(p.getProgPassesP90())).setHeader("Prog Pass/90").setWidth(WIDTH_120).setSortable(true);
            grid.addColumn(p -> fmt(p.getTouchesP90())).setHeader("Touches/90").setWidth(WIDTH_110).setSortable(true);
        });
        
        columnConfigurations.put(PER90_DEF, () -> {
            grid.addColumn(p -> fmt(p.getTacklesP90())).setHeader("Tkl/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getInterceptionsP90())).setHeader("Int/90").setWidth(WIDTH_90).setSortable(true);
            grid.addColumn(p -> fmt(p.getClearancesP90())).setHeader("Clear/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getBlocksP90())).setHeader("Blocks/90").setWidth(WIDTH_100).setSortable(true);
            grid.addColumn(p -> fmt(p.getRecoveriesP90())).setHeader("Recov/90").setWidth(WIDTH_110).setSortable(true);
        });
    }
    
    private void configureDefaultColumns() {
        grid.removeAllColumns();
        List.of(BASIC_INFO, PLAYING_TIME, STANDARD_ATTACKING, SHOOTING, 
                EXPECTED_METRICS, PASSING, PROGRESSIVE_PLAY, 
                POSSESSION, DEFENSIVE, GOAL_SHOT_CREATION, 
                PER90_ATT, PER90_GEN, PER90_DEF)
            .forEach(key -> {
                if(columnConfigurations.containsKey(key)) {
                    columnConfigurations.get(key).run();
                }
            });
    }
    
    private void updateColumnVisibility() {
        grid.removeAllColumns();
        var selected = columnSelector.getSelectedItems();
        
        if (selected.isEmpty()) {
            configureDefaultColumns();
        } else {
            columnConfigurations.forEach((key, config) -> {
                if (selected.contains(key)) {
                    config.run();
                }
            });
        }
    }
}
package com.scouting.UI;

import com.scouting.data.model.Player;
import com.scouting.service.ScoutingService;
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
import com.vaadin.flow.component.orderedlayout.FlexComponent.Alignment;
import com.vaadin.flow.component.orderedlayout.FlexComponent.JustifyContentMode;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.Scroller;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.data.value.ValueChangeMode;
import com.vaadin.flow.router.Route;

import java.util.*;
import java.util.stream.Collectors;

@Route("")
public class MainView extends VerticalLayout {
    
    private final ScoutingService scoutingService;
    private final Grid<Player> grid;
    private final Span recordCount;
    private final List<Player> allPlayers;
    
    private IntegerField minAgeField;
    private IntegerField maxAgeField;
    private TextField playerSearch;
    private TextField squadSearch;
    private ComboBox<String> compFilter;
    private ComboBox<String> nationFilter;
    private ComboBox<String> positionFilter;
    
    private MultiSelectComboBox<String> columnSelector;
    private Map<String, Runnable> columnConfigurations;

    private VerticalLayout filterRowsLayout; 
    private List<FilterRowComponent> activeFilterRows; 
    private Map<String, String> statFieldMapping; 

    public MainView(ScoutingService scoutingService) {
        this.scoutingService = scoutingService;
        this.allPlayers = scoutingService.getAllPlayers();
        
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
        initializeStatMapping();
        initializeColumnConfigurations();
        
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
    
    private void initializeStatMapping() {
        statFieldMapping = new LinkedHashMap<>();
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
        columnSelector = new MultiSelectComboBox<>("Seleziona Metriche (US-03)");
        columnSelector.setItems(columnConfigurations.keySet());
        columnSelector.select("Basic Info"); 
        columnSelector.setWidth("100%");
        columnSelector.setMaxWidth("600px");
        columnSelector.setPlaceholder("Scegli categorie statistiche...");
        columnSelector.addValueChangeListener(e -> updateColumnVisibility());
    }
    
    private void createFilters() {
        minAgeField = new IntegerField("Min Age");
        minAgeField.setValue(16); minAgeField.setWidth("80px");
        
        maxAgeField = new IntegerField("Max Age");
        maxAgeField.setValue(50); maxAgeField.setWidth("80px");
        
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
        positionFilter.setWidth("110px");
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
        scroller.getStyle().set("max-height", "150px"); 
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
            (componentToRemove) -> { 
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

        List<Player> filtered = scoutingService.findPlayersByCriteria(
            minAge, maxAge, playerName, squad, comp, nation, position,
            statCriteriaList
        );

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
        
        columnSelector.select("Basic Info");
        
        updateList();
    }
    
    private void updateRecordCount() {
        recordCount.setText("Visualizzando %d giocatori (Database 2025-26)".formatted(allPlayers.size()));
    }
    
    private String fmt(Double value) {
        return value != null ? "%.2f".formatted(value) : "-";
    }
    
    
    private void initializeColumnConfigurations() {
        columnConfigurations = new LinkedHashMap<>();

        columnConfigurations.put("Basic Info", () -> {
            grid.addColumn(Player::getName).setHeader("Player").setFrozen(true).setWidth("180px").setSortable(true).setResizable(true);
            grid.addColumn(Player::getAge).setHeader("Age").setWidth("70px").setSortable(true);
            grid.addColumn(Player::getPosition).setHeader("Position").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getSquad).setHeader("Squad").setWidth("150px").setSortable(true).setResizable(true);
            grid.addColumn(Player::getNation).setHeader("Nation").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getCompetition).setHeader("Competition").setWidth("150px").setSortable(true).setResizable(true);
        });

        columnConfigurations.put("Playing Time", () -> {
            grid.addColumn(Player::getMatches).setHeader("Matches").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getStarts).setHeader("Starts").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getMinutes).setHeader("Minutes").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getNinetyS())).setHeader("90s").setWidth("80px").setSortable(true);
        });

        columnConfigurations.put("Standard Attacking", () -> {
            grid.addColumn(Player::getGoals).setHeader("Goals").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getAssists).setHeader("Assists").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getgPlusA).setHeader("G+A").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getNpg).setHeader("Non-Pen Goals").setWidth("130px").setSortable(true);
            grid.addColumn(Player::getPkMade).setHeader("PK Made").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getPkAttempted).setHeader("PK Att").setWidth("100px").setSortable(true);
        });

        columnConfigurations.put("Shooting", () -> {
            grid.addColumn(Player::getShots).setHeader("Shots").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getSot).setHeader("SoT").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getSotPercentage())).setHeader("SoT%").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getgPerShots())).setHeader("G/Sh").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getgPerSot())).setHeader("G/SoT").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getAvgShotDist())).setHeader("Avg Dist").setWidth("110px").setSortable(true);
        });

        columnConfigurations.put("Expected Metrics (xG)", () -> {
            grid.addColumn(p -> fmt(p.getXg())).setHeader("xG").setWidth("80px").setSortable(true);
            grid.addColumn(p -> fmt(p.getNpxg())).setHeader("npxG").setWidth("80px").setSortable(true);
            grid.addColumn(p -> fmt(p.getXag())).setHeader("xAG").setWidth("80px").setSortable(true);
            grid.addColumn(p -> fmt(p.getNpxgPlusXag())).setHeader("npxG+xAG").setWidth("110px").setSortable(true);
            grid.addColumn(p -> fmt(p.getgMinusXg())).setHeader("G-xG").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getNpgMinusXg())).setHeader("npG-xG").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getaMinusXag())).setHeader("A-xAG").setWidth("90px").setSortable(true);
        });
        
        columnConfigurations.put("Passing", () -> {
            grid.addColumn(Player::getPassesCompleted).setHeader("Pass Cmp").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getPassesAttempted).setHeader("Pass Att").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getPassesPercentage())).setHeader("Pass %").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getKeyPasses).setHeader("Key Passes").setWidth("110px").setSortable(true);
            grid.addColumn(Player::getPassesFinalThird).setHeader("Pass Final 3rd").setWidth("130px").setSortable(true);
            grid.addColumn(Player::getPassesPenArea).setHeader("Pass Pen Area").setWidth("130px").setSortable(true);
            grid.addColumn(Player::getCrossesPenArea).setHeader("Cross Pen Area").setWidth("130px").setSortable(true);
        });
        
        columnConfigurations.put("Progressive Play", () -> {
            grid.addColumn(Player::getProgCarries).setHeader("Prog Carries").setWidth("120px").setSortable(true);
            grid.addColumn(Player::getProgPasses).setHeader("Prog Passes").setWidth("120px").setSortable(true);
            grid.addColumn(Player::getProgPassesReceived).setHeader("Prog Rec").setWidth("120px").setSortable(true);
        });
        
        columnConfigurations.put("Possession & Dribbling", () -> {
            grid.addColumn(Player::getTouches).setHeader("Touches").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getTakeOnAttempted).setHeader("Dribble Att").setWidth("110px").setSortable(true);
            grid.addColumn(Player::getTakeOnSucc).setHeader("Dribble Succ").setWidth("110px").setSortable(true);
            grid.addColumn(p -> fmt(p.getTakeOnPercentage())).setHeader("Dribble %").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getCarries).setHeader("Carries").setWidth("100px").setSortable(true);
            grid.addColumn(Player::getDispossessed).setHeader("Dispossessed").setWidth("120px").setSortable(true);
            grid.addColumn(Player::getMiscontrols).setHeader("Miscontrols").setWidth("110px").setSortable(true);
        });

        columnConfigurations.put("Defensive Actions", () -> {
            grid.addColumn(Player::getTackles).setHeader("Tackles").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getTacklesWon).setHeader("Tkl Won").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getTacklesPercentage())).setHeader("Tkl %").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getInterceptions).setHeader("Int").setWidth("80px").setSortable(true);
            grid.addColumn(Player::getTklPlusInt).setHeader("Tkl+Int").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getBlocks).setHeader("Blocks").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getClearances).setHeader("Clear").setWidth("90px").setSortable(true);
            grid.addColumn(Player::getErrorsLeadingShot).setHeader("Errors").setWidth("90px").setSortable(true);
        });

        columnConfigurations.put("Goal & Shot Creation", () -> {
            grid.addColumn(Player::getSca).setHeader("SCA").setWidth("80px").setSortable(true);
            grid.addColumn(Player::getGca).setHeader("GCA").setWidth("80px").setSortable(true);
        });

        columnConfigurations.put("Per 90 (Attacking)", () -> {
            grid.addColumn(p -> fmt(p.getGoalsP90())).setHeader("Goals/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getAssistsP90())).setHeader("Assists/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getgPlusAP90())).setHeader("G+A/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getNpgP90())).setHeader("npg/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getXgP90())).setHeader("xG/90").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getXagP90())).setHeader("xAG/90").setWidth("90px").setSortable(true);
        });

        columnConfigurations.put("Per 90 (General Play)", () -> {
            grid.addColumn(p -> fmt(p.getShotsP90())).setHeader("Shots/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getSotP90())).setHeader("SoT/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getKeyPassesP90())).setHeader("Key Pass/90").setWidth("120px").setSortable(true);
            grid.addColumn(p -> fmt(p.getProgCarriesP90())).setHeader("Prog Carr/90").setWidth("120px").setSortable(true);
            grid.addColumn(p -> fmt(p.getProgPassesP90())).setHeader("Prog Pass/90").setWidth("120px").setSortable(true);
            grid.addColumn(p -> fmt(p.getTouchesP90())).setHeader("Touches/90").setWidth("110px").setSortable(true);
        });
        
        columnConfigurations.put("Per 90 (Defensive)", () -> {
            grid.addColumn(p -> fmt(p.getTacklesP90())).setHeader("Tkl/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getInterceptionsP90())).setHeader("Int/90").setWidth("90px").setSortable(true);
            grid.addColumn(p -> fmt(p.getClearancesP90())).setHeader("Clear/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getBlocksP90())).setHeader("Blocks/90").setWidth("100px").setSortable(true);
            grid.addColumn(p -> fmt(p.getRecoveriesP90())).setHeader("Recov/90").setWidth("110px").setSortable(true);
        });
    }
    
    private void configureDefaultColumns() {
        grid.removeAllColumns();
        List.of("Basic Info", "Playing Time", "Standard Attacking", "Shooting", 
                "Expected Metrics (xG)", "Passing", "Progressive Play", 
                "Possession & Dribbling", "Defensive Actions", "Goal & Shot Creation", 
                "Per 90 (Attacking)", "Per 90 (General Play)", "Per 90 (Defensive)")
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
package com.scouting.UI;

import com.scouting.data.model.Player;
import com.scouting.service.ScoutingService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.combobox.MultiSelectComboBox;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
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
    
    private IntegerField minAgeField, maxAgeField;
    private TextField playerSearch, squadSearch;
    private ComboBox<String> compFilter, nationFilter, positionFilter;
    private MultiSelectComboBox<String> columnSelector;
    
    private ComboBox<String> statSelector;
    private TextField minStatValue, maxStatValue;
    private Map<String, String> statFieldMapping;
    private Map<String, Runnable> columnConfigurations;
    
    public MainView(ScoutingService scoutingService) {
        this.scoutingService = scoutingService;
        this.allPlayers = scoutingService.getAllPlayers();
        
        setSizeFull();
        setPadding(true);
        
        add(new H1("Football Player Scouting Suite"));
        
        grid = new Grid<>(Player.class, false);
        initializeColumnConfigurations();
        configureDefaultColumns();
        
        createFilters();
        
        recordCount = new Span();
        recordCount.getStyle().set("font-weight", "bold");
        
        add(recordCount, grid);
        updateList();
    }
    
    private void createFilters() {
        initializeStatFilter();
        
        minAgeField = createIntField("Min Age", 16);
        maxAgeField = createIntField("Max Age", 50);
        
        playerSearch = createSearchField("Player Name", "Search player...");
        squadSearch = createSearchField("Squad", "Search squad...");
        
        compFilter = createCombo("Competition", Player::getCompetition);
        nationFilter = createCombo("Nation", Player::getNation);
        positionFilter = createCombo("Position", Player::getPosition);
        
        columnSelector = new MultiSelectComboBox<>("Visible Columns");
        columnSelector.setItems(columnConfigurations.keySet());
        columnSelector.select("Basic Info");
        columnSelector.setWidth("300px");
        columnSelector.addValueChangeListener(e -> updateColumnVisibility());
        
        var resetBtn = new Button("Reset Filters", e -> resetFilters());
        
        var filters1 = new HorizontalLayout(minAgeField, maxAgeField, playerSearch, squadSearch);
        var filters2 = new HorizontalLayout(compFilter, nationFilter, positionFilter, columnSelector);
        var statFilterLayout = new HorizontalLayout(statSelector, minStatValue, maxStatValue);
        
        filters1.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        filters2.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        statFilterLayout.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        
        var bottomBar = new HorizontalLayout(statFilterLayout, resetBtn);
        bottomBar.setWidthFull();
        bottomBar.setJustifyContentMode(JustifyContentMode.BETWEEN);
        
        add(filters1, filters2, bottomBar);
    }
    
    private IntegerField createIntField(String label, int defaultVal) {
        var field = new IntegerField(label);
        field.setValue(defaultVal);
        field.setWidth("120px");
        field.addValueChangeListener(e -> updateList());
        return field;
    }

    private TextField createSearchField(String label, String placeholder) {
        var field = new TextField(label);
        field.setPlaceholder(placeholder);
        field.setWidth("200px");
        field.setValueChangeMode(ValueChangeMode.LAZY);
        field.addValueChangeListener(e -> updateList());
        return field;
    }

    private ComboBox<String> createCombo(String label, java.util.function.Function<Player, String> mapper) {
        var field = new ComboBox<String>(label);
        var items = allPlayers.stream()
                .map(mapper)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        field.setItems(items);
        field.setWidth("200px");
        field.setClearButtonVisible(true);
        field.addValueChangeListener(e -> updateList());
        return field;
    }
    
    private void initializeColumnConfigurations() {
        columnConfigurations = new LinkedHashMap<>();

        columnConfigurations.put("Basic Info", () -> {
            grid.addColumn(Player::getName).setHeader("Player").setFrozen(true).setWidth("180px");
            grid.addColumn(Player::getAge).setHeader("Age").setWidth("70px");
            grid.addColumn(Player::getPosition).setHeader("Position").setWidth("90px");
            grid.addColumn(Player::getSquad).setHeader("Squad").setWidth("150px");
            grid.addColumn(Player::getNation).setHeader("Nation").setWidth("100px");
            grid.addColumn(Player::getCompetition).setHeader("Competition").setWidth("150px");
        });

        columnConfigurations.put("Playing Time", () -> {
            grid.addColumn(Player::getMatches).setHeader("Matches").setWidth("80px");
            grid.addColumn(Player::getStarts).setHeader("Starts").setWidth("80px");
            grid.addColumn(Player::getMinutes).setHeader("Minutes").setWidth("90px");
            grid.addColumn(p -> fmt(p.getNinetyS())).setHeader("90s").setWidth("70px");
        });

        columnConfigurations.put("Standard Attacking", () -> {
            grid.addColumn(Player::getGoals).setHeader("Goals").setWidth("80px");
            grid.addColumn(Player::getAssists).setHeader("Assists").setWidth("80px");
            grid.addColumn(Player::getgPlusA).setHeader("G+A").setWidth("80px");
            grid.addColumn(Player::getNpg).setHeader("Non-Penalty Goals").setWidth("140px");
            grid.addColumn(Player::getPkMade).setHeader("PK Made").setWidth("90px");
            grid.addColumn(Player::getPkAttempted).setHeader("PK Att").setWidth("90px");
        });

        columnConfigurations.put("Shooting", () -> {
            grid.addColumn(Player::getShots).setHeader("Shots").setWidth("80px");
            grid.addColumn(Player::getSot).setHeader("SoT").setWidth("80px");
            grid.addColumn(p -> fmt(p.getSotPercentage())).setHeader("SoT%").setWidth("80px");
            grid.addColumn(p -> fmt(p.getgPerShots())).setHeader("G/Sh").setWidth("80px");
            grid.addColumn(p -> fmt(p.getgPerSot())).setHeader("G/SoT").setWidth("80px");
            grid.addColumn(p -> fmt(p.getAvgShotDist())).setHeader("Avg Shot Dist").setWidth("120px");
        });

        columnConfigurations.put("Expected Metrics (xG)", () -> {
            grid.addColumn(p -> fmt(p.getXg())).setHeader("xG").setWidth("70px");
            grid.addColumn(p -> fmt(p.getNpxg())).setHeader("npxG").setWidth("70px");
            grid.addColumn(p -> fmt(p.getXag())).setHeader("xAG").setWidth("70px");
            grid.addColumn(p -> fmt(p.getNpxgPlusXag())).setHeader("npxG+xAG").setWidth("100px");
            grid.addColumn(p -> fmt(p.getgMinusXg())).setHeader("G-xG").setWidth("80px");
            grid.addColumn(p -> fmt(p.getNpgMinusXg())).setHeader("npG-xG").setWidth("90px");
            grid.addColumn(p -> fmt(p.getaMinusXag())).setHeader("A-xAG").setWidth("80px");
        });
        
        columnConfigurations.put("Passing", () -> {
            grid.addColumn(Player::getPassesCompleted).setHeader("Pass Cmp").setWidth("90px");
            grid.addColumn(Player::getPassesAttempted).setHeader("Pass Att").setWidth("90px");
            grid.addColumn(p -> fmt(p.getPassesPercentage())).setHeader("Pass %").setWidth("80px");
            grid.addColumn(Player::getKeyPasses).setHeader("Key Passes").setWidth("100px");
            grid.addColumn(Player::getPassesFinalThird).setHeader("Passes Final 3rd").setWidth("130px");
            grid.addColumn(Player::getPassesPenArea).setHeader("Passes Pen Area").setWidth("130px");
            grid.addColumn(Player::getCrossesPenArea).setHeader("Crosses Pen Area").setWidth("140px");
        });
        
        columnConfigurations.put("Progressive Play", () -> {
            grid.addColumn(Player::getProgCarries).setHeader("Prog Carries").setWidth("110px");
            grid.addColumn(Player::getProgPasses).setHeader("Prog Passes").setWidth("110px");
            grid.addColumn(Player::getProgPassesReceived).setHeader("Prog Pass Rec").setWidth("120px");
        });
        
        columnConfigurations.put("Possession & Dribbling", () -> {
            grid.addColumn(Player::getTouches).setHeader("Touches").setWidth("90px");
            grid.addColumn(Player::getTakeOnAttempted).setHeader("Dribble Att").setWidth("100px");
            grid.addColumn(Player::getTakeOnSucc).setHeader("Dribble Succ").setWidth("100px");
            grid.addColumn(p -> fmt(p.getTakeOnPercentage())).setHeader("Dribble %").setWidth("90px");
            grid.addColumn(Player::getCarries).setHeader("Carries").setWidth("90px");
            grid.addColumn(Player::getDispossessed).setHeader("Dispossessed").setWidth("110px");
            grid.addColumn(Player::getMiscontrols).setHeader("Miscontrols").setWidth("100px");
        });

        columnConfigurations.put("Defensive Actions", () -> {
            grid.addColumn(Player::getTackles).setHeader("Tackles").setWidth("90px");
            grid.addColumn(Player::getTacklesWon).setHeader("Tackles Won").setWidth("110px");
            grid.addColumn(p -> fmt(p.getTacklesPercentage())).setHeader("Tackle %").setWidth("90px");
            grid.addColumn(Player::getInterceptions).setHeader("Interceptions").setWidth("110px");
            grid.addColumn(Player::getTklPlusInt).setHeader("Tkl+Int").setWidth("90px");
            grid.addColumn(Player::getBlocks).setHeader("Blocks").setWidth("80px");
            grid.addColumn(Player::getClearances).setHeader("Clearances").setWidth("100px");
            grid.addColumn(Player::getErrorsLeadingShot).setHeader("Errors").setWidth("80px");
        });

        columnConfigurations.put("Goal & Shot Creation", () -> {
            grid.addColumn(Player::getSca).setHeader("SCA").setWidth("70px");
            grid.addColumn(Player::getGca).setHeader("GCA").setWidth("70px");
        });

        columnConfigurations.put("Per 90 (Attacking)", () -> {
            grid.addColumn(p -> fmt(p.getGoalsP90())).setHeader("Goals/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getAssistsP90())).setHeader("Assists/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getgPlusAP90())).setHeader("G+A/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getNpgP90())).setHeader("npg/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getXgP90())).setHeader("xG/90").setWidth("80px");
            grid.addColumn(p -> fmt(p.getXagP90())).setHeader("xAG/90").setWidth("80px");
        });

        columnConfigurations.put("Per 90 (General Play)", () -> {
            grid.addColumn(p -> fmt(p.getShotsP90())).setHeader("Shots/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getSotP90())).setHeader("SoT/90").setWidth("80px");
            grid.addColumn(p -> fmt(p.getKeyPassesP90())).setHeader("Key Pass/90").setWidth("110px");
            grid.addColumn(p -> fmt(p.getProgCarriesP90())).setHeader("Prog Carr/90").setWidth("110px");
            grid.addColumn(p -> fmt(p.getProgPassesP90())).setHeader("Prog Pass/90").setWidth("110px");
            grid.addColumn(p -> fmt(p.getTouchesP90())).setHeader("Touches/90").setWidth("100px");
        });
        
        columnConfigurations.put("Per 90 (Defensive)", () -> {
            grid.addColumn(p -> fmt(p.getTacklesP90())).setHeader("Tackles/90").setWidth("100px");
            grid.addColumn(p -> fmt(p.getInterceptionsP90())).setHeader("Int/90").setWidth("80px");
            grid.addColumn(p -> fmt(p.getClearancesP90())).setHeader("Clear/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getBlocksP90())).setHeader("Blocks/90").setWidth("90px");
            grid.addColumn(p -> fmt(p.getRecoveriesP90())).setHeader("Recoveries/90").setWidth("120px");
        });
    }
    
    private void configureDefaultColumns() {
        grid.removeAllColumns();
        List.of("Basic Info", "Playing Time", "Standard Attacking", "Shooting", 
                "Expected Metrics (xG)", "Passing", "Progressive Play", 
                "Possession & Dribbling", "Defensive Actions", "Goal & Shot Creation", 
                "Per 90 (Attacking)", "Per 90 (General Play)", "Per 90 (Defensive)")
            .forEach(key -> columnConfigurations.get(key).run());
        grid.setSizeFull();
    }
    
    private void updateColumnVisibility() {
        grid.removeAllColumns();
        var selected = columnSelector.getSelectedItems();
        if (selected.isEmpty()) configureDefaultColumns();
        else selected.forEach(key -> columnConfigurations.get(key).run());
        grid.setSizeFull();
    }
    
    private void updateList() {
        var minStat = parseDoubleSafe(minStatValue.getValue());
        var maxStat = parseDoubleSafe(maxStatValue.getValue());
        
        var selectedStatName = statSelector.getValue();
        var statField = selectedStatName != null ? statFieldMapping.get(selectedStatName) : null;

        var filtered = scoutingService.findPlayersByCriteria(
            minAgeField.getValue(), maxAgeField.getValue(),
            playerSearch.getValue(), squadSearch.getValue(),
            compFilter.getValue(), nationFilter.getValue(), positionFilter.getValue(),
            statField, minStat, maxStat
        );

        grid.setItems(filtered);
        recordCount.setText("Showing %d of %d players".formatted(filtered.size(), allPlayers.size()));
    }
    
    private void resetFilters() {
        minAgeField.setValue(16);
        maxAgeField.setValue(50);
        playerSearch.clear();
        squadSearch.clear();
        compFilter.clear();
        nationFilter.clear();
        positionFilter.clear();
        statSelector.clear();
        minStatValue.clear();
        maxStatValue.clear();
        columnSelector.select("Basic Info");
        updateList();
    }
    
    private String fmt(Double value) {
        return value != null ? "%.2f".formatted(value) : "-";
    }
    
    private Double parseDoubleSafe(String val) {
        try { return Double.parseDouble(val); } catch (Exception e) { return null; }
    }
    
    private void initializeStatFilter() {
        statFieldMapping = Map.ofEntries(
            Map.entry("Goals", "goals"),
            Map.entry("Assists", "assists"),
            Map.entry("xG (Expected Goals)", "xg"),
            Map.entry("xAG (Expected Assists)", "xag"),
            Map.entry("Shots", "shots"),
            Map.entry("SoT (Shots on Target)", "sot"),
            Map.entry("Progressive Passes", "progPasses"),
            Map.entry("Progressive Carries", "progCarries"),
            Map.entry("Tackles", "tackles"),
            Map.entry("Interceptions", "interceptions"),
            Map.entry("Goals per 90", "goalsP90"),
            Map.entry("Assists per 90", "assistsP90"),
            Map.entry("xG per 90", "xgP90"),
            Map.entry("Tackles per 90", "tacklesP90")
        );
        
        statSelector = new ComboBox<>("Add Filter");
        statSelector.setItems(statFieldMapping.keySet());
        statSelector.setWidth("250px");
        statSelector.setClearButtonVisible(true);
        statSelector.addValueChangeListener(e -> updateList());

        minStatValue = new TextField("Min");
        minStatValue.setWidth("100px");
        minStatValue.setValueChangeMode(ValueChangeMode.LAZY);
        minStatValue.addValueChangeListener(e -> updateList());

        maxStatValue = new TextField("Max");
        maxStatValue.setWidth("100px");
        maxStatValue.setValueChangeMode(ValueChangeMode.LAZY);
        maxStatValue.addValueChangeListener(e -> updateList());
    }
}